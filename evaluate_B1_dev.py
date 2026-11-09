from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse
import csv
import json

import joblib
import numpy as np
import sklearn

from check_B1_stability import evaluate, LOWER, UPPER, save_json
from text_features import tokenize_review


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels",
        type=Path,
        help="Final manually labelled CSV",
    )
    args = parser.parse_args()
    labels_path = args.labels.resolve()

    # 1. Seçilen modelin ve eşiklerin korunduğunu kontrol et.
    stability = json.loads(
        (REPORTS / "B1_threshold_stability.json").read_text(
            encoding="utf-8"
        )
    )

    if stability["outcome"] not in (
        "KEEP_B1",
        "KEEP_B1_WITH_TOLERANCE",
    ):
        raise ValueError(
            "B1 has not passed the agreed selection rule."
        )

    plan = stability["plan"]

    if (
        (LOWER, UPPER) != (0.50, 0.65)
        or (plan["lower"], plan["upper"]) != (LOWER, UPPER)
    ):
        raise ValueError("Decision thresholds have changed.")

    training_path = DATA / "training_B1_v3.csv"

    if digest(training_path) != plan["training_csv_sha256"]:
        raise ValueError(
            "Training data changed after stability evaluation."
        )

    with training_path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        train = list(csv.DictReader(file, delimiter=";"))

    train_ids = {row["review_id"] for row in train}
    train_keys = {
        tuple(tokenize_review(row["content"]))
        for row in train
    }

    # 2. Dev/test ayrımını manifestten oku.
    manifest_path = (
        DATA / "evaluation_v1" / "selection_manifest.json"
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    selections = manifest["selections"]

    if Counter(row["split"] for row in selections) != {
        "dev": 30,
        "test": 100,
    }:
        raise ValueError("Unexpected development/test split.")

    manifest_ids = {
        row["review_id"] for row in selections
    }

    if len(manifest_ids) != len(selections):
        raise ValueError("Duplicate IDs in selection manifest.")

    for selection in selections:
        held_ids = set(selection["group_review_ids"])
        held_ids.add(selection["review_id"])

        if (
            held_ids & train_ids
            or tuple(selection["group_tokens"]) in train_keys
        ):
            raise ValueError(
                "Evaluation/training overlap detected."
            )

    # 3. Etiket dosyasının sütunlarını ve kimliklerini doğrula.
    with labels_path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        delimiter = csv.Sniffer().sniff(
            file.readline(),
            delimiters=";,\t",
        ).delimiter

        file.seek(0)
        reader = csv.DictReader(file, delimiter=delimiter)

        required = {"review_id", "content", "sentiment"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Expected review_id, content and sentiment columns."
            )

        rows = list(reader)

    by_id = {
        row["review_id"]: row
        for row in rows
    }

    if (
        len(by_id) != len(rows)
        or set(by_id) != manifest_ids
    ):
        raise ValueError(
            "Label CSV IDs do not match the 130 selected comments."
        )

    # Yalnızca development içeriklerini ve etiketlerini seç.
    dev = []
    dev_keys = set()

    for selection in selections:
        if selection["split"] != "dev":
            continue

        row = dict(by_id[selection["review_id"]])
        row["sentiment"] = row["sentiment"].strip().lower()

        if row["sentiment"] not in {
            "negative", "positive", "neutral", "unclear",
        }:
            raise ValueError(
                f"Invalid development label: {row['review_id']}"
            )

        key = tuple(tokenize_review(row["content"]))

        if (
            key != tuple(selection["group_tokens"])
            or key in dev_keys
        ):
            raise ValueError(
                "Changed or duplicate development group: "
                f"{row['review_id']}"
            )

        dev_keys.add(key)
        dev.append(row)

    # 4. Eğitilmiş B1 modelini yükle; fit() çağrısı yok.
    model_path = ROOT / "models" / "sentiment_B1.joblib"
    model_hash = digest(model_path)
    model = joblib.load(model_path)

    if set(model.classes_) != {"negative", "positive"}:
        raise ValueError("Expected a binary B1 model.")

    if model.named_steps["classifier"].class_weight is not None:
        raise ValueError("Unexpected B1 class weighting.")

    # Eğitimdeki gibi model kopyasında ten rengini eşitle.
    texts = [
        "".join(
            character
            for character in row["content"]
            if not 0x1F3FB <= ord(character) <= 0x1F3FF
        )
        for row in dev
    ]

    positive_column = list(model.classes_).index("positive")
    probability = model.predict_proba(texts)[:, positive_column]

    if (
        not np.all(np.isfinite(probability))
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Invalid model probabilities.")

    actual = np.array([
        row["sentiment"] for row in dev
    ])

    decisions = np.where(
        probability < LOWER,
        "negative",
        np.where(
            probability >= UPPER,
            "positive",
            "needs_review",
        ),
    )

    # 5. İki sınıflı başarıyı yalnızca pozitif/negatiflerde ölç.
    binary = np.isin(actual, ["negative", "positive"])

    if not {"negative", "positive"}.issubset(
        set(actual[binary])
    ):
        raise ValueError(
            "Both sentiment classes are needed for this dev report."
        )

    metrics, counts, _, _ = evaluate(
        actual[binary],
        probability[binary],
    )

    if not np.any(decisions[binary] != "needs_review"):
        metrics["accuracy"] = None
        metrics["accepted_macro_f1"] = None

    all_dev_coverage = float(
        np.mean(decisions != "needs_review")
    )

    print(
        "Development labels:",
        dict(Counter(actual.tolist())),
    )
    print("All development records:", len(dev))
    print("Binary development records:", int(binary.sum()))
    print(
        f"All-dev automatic coverage: {all_dev_coverage:.4f}"
    )

    print(
        "\nBinary-only metrics "
        "(neutral/unclear reported separately):"
    )

    for name, value in metrics.items():
        if value is None:
            print(f"{name}: undefined")
        else:
            print(f"{name}: {value:.4f}")

    print("Counts:", counts)

    # Nötr ve belirsiz yorumların kararlarını ayrıca göster.
    print("\nDecision counts by manual label:")

    for label in (
        "negative", "positive", "neutral", "unclear"
    ):
        print(
            label,
            dict(Counter(decisions[actual == label].tolist())),
        )

    predictions = []

    print(
        "\nErrors, review decisions "
        "and neutral/unclear comments:"
    )

    for index, row in enumerate(dev):
        record = {
            "review_id": row["review_id"],
            "content": row["content"],
            "actual": row["sentiment"],
            "decision": str(decisions[index]),
            "positive_probability": float(probability[index]),
        }
        predictions.append(record)

        if decisions[index] != actual[index]:
            print(f"\nID: {row['review_id']}")
            print(
                f"Actual: {actual[index]} | "
                f"Decision: {decisions[index]} | "
                f"P(positive): {probability[index]:.4f}"
            )
            print(row["content"])

    # 6. Kullanılan modelin kimliğiyle birlikte raporu kaydet.
    report_path = REPORTS / "B1_dev.json"

    save_json(
        report_path,
        {
            "experiment": "B1_dev",
            "model_sha256": model_hash,
            "labels_csv_sha256": digest(labels_path),
            "manifest_sha256": digest(manifest_path),
            "tokenizer_sha256": digest(
                ROOT / "text_features.py"
            ),
            "training_csv_sha256": digest(training_path),
            "sklearn_version": sklearn.__version__,
            "lower": LOWER,
            "upper": UPPER,
            "all_dev_coverage": all_dev_coverage,
            "binary_metrics": metrics,
            "binary_counts": counts,
            "predictions": predictions,
            "final_test_scored": False,
        },
    )

    print("\nSaved:", report_path)
    print(
        "Final test predictions: not generated. "
        "Model and thresholds: unchanged."
    )


if __name__ == "__main__":
    main()