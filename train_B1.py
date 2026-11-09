import csv
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
LABELS = ["negative", "positive"]


def main():
    csv_path = DATA / "training_B1_v3.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))

    ids = [row["review_id"] for row in rows]
    texts = np.array([row["content"] for row in rows], dtype=object)
    labels = np.array([row["sentiment"] for row in rows])
    groups = np.array([int(row["group_id"]) for row in rows])

    if len(set(ids)) != len(ids) or Counter(labels) != {
        "positive": 1947, "negative": 530,
    }:
        raise ValueError("Unexpected review IDs or B1 class counts.")

    manifest_path = DATA / "evaluation_v1" / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    held_ids = set()
    for row in manifest["selections"]:
        held_ids.add(row["review_id"])
        held_ids.update(row["group_review_ids"])

    if set(ids) & held_ids:
        raise ValueError("Development or test reviews found in B1 training.")

    group_labels = {}
    for group, label in zip(groups, labels):
        if group_labels.setdefault(group, label) != label:
            raise ValueError(f"Conflicting labels for group {group}.")

    unique_groups = np.array(sorted(group_labels))
    unique_labels = [group_labels[group] for group in unique_groups]

    if Counter(unique_labels) != {"positive": 780, "negative": 518}:
        raise ValueError("Unexpected B1 group counts.")

    # clone() copies A1's settings. Each fold learns its own vocabulary.
    template = joblib.load(ROOT / "models" / "sentiment_A1.joblib")
    if template.named_steps["classifier"].class_weight is not None:
        raise ValueError("Expected the original A1 without class weights.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted = np.empty(len(rows), dtype=object)
    majority = np.empty(len(rows), dtype=object)
    probability = np.empty(len(rows))
    coverage = np.zeros(len(rows), dtype=int)

    # Give each text group a total weight of one.
    sizes = Counter(groups)
    weights = np.array([1 / sizes[group] for group in groups])
    folds = []

    print(f"Training: {len(rows)} records, {len(unique_groups)} groups")

    for fold, (fit_groups, val_groups) in enumerate(
        cv.split(unique_groups, unique_labels), start=1
    ):
        fit = np.isin(groups, unique_groups[fit_groups])
        val = np.isin(groups, unique_groups[val_groups])

        model = clone(template)
        model.fit(texts[fit], labels[fit])
        predicted[val] = model.predict(texts[val])

        positive_column = list(model.classes_).index("positive")
        probability[val] = model.predict_proba(texts[val])[:, positive_column]
        majority[val] = Counter(labels[fit]).most_common(1)[0][0]
        coverage[val] += 1

        score = f1_score(
            labels[val], predicted[val], labels=LABELS,
            average="macro", sample_weight=weights[val],
        )
        folds.append({
            "fold": fold,
            "validation_review_ids": [ids[i] for i in np.flatnonzero(val)],
            "group_macro_f1": float(score),
        })
        print(f"Fold {fold}: {len(val_groups)} groups, "
              f"{val.sum()} records, group macro-F1={score:.4f}")

    if not np.all(coverage == 1):
        raise ValueError("Every record must have exactly one OOF prediction.")

    record_f1 = f1_score(labels, predicted, labels=LABELS, average="macro")
    group_f1 = f1_score(
        labels, predicted, labels=LABELS, average="macro",
        sample_weight=weights,
    )
    majority_f1 = f1_score(
        labels, majority, labels=LABELS, average="macro",
        sample_weight=weights,
    )

    print(f"OOF record macro-F1: {record_f1:.4f}")
    print(f"OOF group macro-F1: {group_f1:.4f}")
    print(f"Majority group macro-F1: {majority_f1:.4f}")
    print(classification_report(
        labels, predicted, labels=LABELS, sample_weight=weights,
        digits=4, zero_division=0,
    ))

    report = {
        "experiment": "B1",
        "split_version": "grouped_v3",
        "training_csv_sha256": sha256(csv_path.read_bytes()).hexdigest(),
        "folds": folds,
        "record_macro_f1": float(record_f1),
        "group_macro_f1": float(group_f1),
        "majority_group_macro_f1": float(majority_f1),
        "predictions": [
            {
                "review_id": ids[i],
                "group_id": int(groups[i]),
                "actual": str(labels[i]),
                "predicted": str(predicted[i]),
                "positive_probability": float(probability[i]),
            }
            for i in range(len(rows))
        ],
    }
    report_path = ROOT / "reports" / "oof_B1_grouped_v3.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    final_model = clone(template)
    final_model.fit(texts, labels)
    model_path = ROOT / "models" / "sentiment_B1.joblib"
    joblib.dump(final_model, model_path)
    print("Saved:", report_path)
    print("Saved:", model_path)


if __name__ == "__main__":
    main()