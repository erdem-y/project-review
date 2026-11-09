from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import csv
import json

import joblib
import numpy as np
from sklearn.metrics import f1_score

from freeze_B1 import digest, read_json
from text_features import tokenize_review


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "B1_frozen.json"
REPORT_PATH = ROOT / "reports" / "B1_test.json"


def binary_metrics(actual, decisions):
    accepted = decisions != "needs_review"
    negative = actual == "negative"

    result = {
        "records": len(actual),
        "automatic_decisions": int(accepted.sum()),
        "correct_automatic_decisions": int(
            np.sum(accepted & (actual == decisions))
        ),
        "coverage": (
            float(accepted.mean()) if len(actual) else None
        ),
        "accuracy": (
            float(np.mean(actual[accepted] == decisions[accepted]))
            if accepted.any()
            else None
        ),
        "accepted_macro_f1": (
            float(
                f1_score(
                    actual[accepted],
                    decisions[accepted],
                    labels=["negative", "positive"],
                    average="macro",
                    zero_division=0,
                )
            )
            if accepted.any()
            else None
        ),
        "negative_coverage": (
            float(accepted[negative].mean())
            if negative.any()
            else None
        ),
    }

    for label in ("negative", "positive"):
        selected = decisions == label
        result[label + "_precision"] = (
            float(np.mean(actual[selected] == label))
            if selected.any()
            else None
        )

    return result


def main():
    # Önceki final test raporunun üzerine yazma.
    if REPORT_PATH.exists():
        raise FileExistsError(
            "B1_test.json already exists. "
            "Keep the original final test report."
        )

    # 1. Sabitlenen ayarları ve dosyaları doğrula.
    config = read_json(CONFIG_PATH)

    if (
        config["model_name"] != "B1"
        or config["schema_version"] != 1
    ):
        raise ValueError("Unexpected frozen configuration.")

    if version("scikit-learn") != config["sklearn_version"]:
        raise ValueError(
            "scikit-learn version differs from "
            "the frozen configuration."
        )

    paths = {}

    for name, info in config["files"].items():
        path = ROOT / info["path"]

        if digest(path) != info["sha256"]:
            raise ValueError(
                f"Frozen file changed: {info['path']}"
            )

        paths[name] = path

    thresholds = config["positive_probability_thresholds"]
    lower = thresholds["lower"]
    upper = thresholds["upper"]

    if (lower, upper) != (0.50, 0.65):
        raise ValueError("Unexpected frozen thresholds.")

    print("Frozen files verified. Thresholds:", lower, upper)

    # 2. Test ayrımını ve eğitimle çakışma olmadığını kontrol et.
    manifest = read_json(paths["selection_manifest"])
    selections = manifest["selections"]

    if Counter(s["split"] for s in selections) != {
        "dev": 30,
        "test": 100,
    }:
        raise ValueError("Unexpected evaluation split.")

    manifest_ids = {
        selection["review_id"]
        for selection in selections
    }

    if len(manifest_ids) != len(selections):
        raise ValueError("Duplicate selection IDs.")

    with paths["training_data"].open(
        encoding="utf-8-sig", newline=""
    ) as file:
        train = list(csv.DictReader(file, delimiter=";"))

    train_ids = {row["review_id"] for row in train}
    train_keys = {
        tuple(tokenize_review(row["content"]))
        for row in train
    }

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

    # 3. Etiketleri oku ve yalnızca test yorumlarını seç.
    with paths["evaluation_labels"].open(
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
            raise ValueError("Required label columns are missing.")

        rows = list(reader)

    by_id = {row["review_id"]: row for row in rows}

    if (
        len(by_id) != len(rows)
        or set(by_id) != manifest_ids
    ):
        raise ValueError(
            "Label IDs do not match the evaluation manifest."
        )

    test = []
    evaluation_keys = set()

    for selection in selections:
        key = tuple(selection["group_tokens"])

        if key in evaluation_keys:
            raise ValueError(
                "Duplicate group in development/test selections."
            )

        evaluation_keys.add(key)

        if selection["split"] != "test":
            continue

        row = dict(by_id[selection["review_id"]])
        row["sentiment"] = row["sentiment"].strip().lower()

        if row["sentiment"] not in {
            "negative", "positive", "neutral", "unclear",
        }:
            raise ValueError(
                f"Invalid test label: {row['review_id']}"
            )

        if tuple(tokenize_review(row["content"])) != key:
            raise ValueError(
                "Test text differs from its selected group: "
                f"{row['review_id']}"
            )

        test.append(row)

    # 4. Sabitlenmiş modeli yükle ve tahmin üret.
    model = joblib.load(paths["model"])

    if set(model.classes_) != {"negative", "positive"}:
        raise ValueError("Expected a binary sentiment model.")

    texts = [
        "".join(
            character
            for character in row["content"]
            if not 0x1F3FB <= ord(character) <= 0x1F3FF
        )
        for row in test
    ]

    positive_column = list(model.classes_).index("positive")
    probability = model.predict_proba(texts)[:, positive_column]

    if (
        not np.all(np.isfinite(probability))
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Invalid prediction probabilities.")

    decisions = np.where(
        probability < lower,
        "negative",
        np.where(
            probability >= upper,
            "positive",
            "needs_review",
        ),
    )

    actual = np.array([
        row["sentiment"] for row in test
    ])

    # 5. Pozitif/negatif başarısını ölç.
    # Nötr ve belirsiz yorumların sonuçlarını ayrıca say.
    binary = np.isin(actual, ["negative", "positive"])

    metrics = binary_metrics(
        actual[binary],
        decisions[binary],
    )

    decision_counts = {
        label: {
            decision: int(
                np.sum(
                    (actual == label)
                    & (decisions == decision)
                )
            )
            for decision in (
                "negative", "positive", "needs_review"
            )
        }
        for label in (
            "negative", "positive", "neutral", "unclear"
        )
    }

    automatic = int(np.sum(decisions != "needs_review"))

    predictions = [
        {
            "review_id": row["review_id"],
            "content": row["content"],
            "actual": row["sentiment"],
            "decision": str(decisions[index]),
            "positive_probability": float(probability[index]),
        }
        for index, row in enumerate(test)
    ]

    # 6. Sonuçları kullanılan config ile birlikte kaydet.
    report = {
        "experiment": "B1_final_test",
        "evaluated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "config_sha256": digest(CONFIG_PATH),
        "evaluation_script_sha256": digest(
            Path(__file__).resolve()
        ),
        "frozen_config": config,
        "test_records": len(test),
        "automatic_decisions_all_test": automatic,
        "all_test_coverage": automatic / len(test),
        "manual_label_counts": dict(
            Counter(actual.tolist())
        ),
        "binary_metrics": metrics,
        "decision_counts": decision_counts,
        "predictions": predictions,
        "final_test_scored": True,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\nFinal test labels:", report["manual_label_counts"])
    print(
        f"All-test automatic coverage: "
        f"{automatic}/{len(test)} = "
        f"{automatic / len(test):.4f}"
    )

    print(
        "\nBinary-only metrics "
        "(neutral/unclear reported separately):"
    )

    for name, value in metrics.items():
        if value is None:
            print(f"{name}: undefined")
        elif isinstance(value, int):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")

    print("\nDecision counts by manual label:")

    for label, counts in decision_counts.items():
        print(label, counts)

    print("\nSaved:", REPORT_PATH)
    print(
        "Final test completed. "
        "Model and thresholds: unchanged."
    )


if __name__ == "__main__":
    main()