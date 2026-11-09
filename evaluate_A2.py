from collections import Counter
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import classification_report

from prepare_training_data import prepare_training_data
from inspect_oof_grouped import (
    LABELS,
    group_weights,
    macro_f1,
    print_report,
)


PROJECT_DIR = Path(__file__).resolve().parent

A1_MODEL_PATH = PROJECT_DIR / "models" / "sentiment_A1.joblib"
A1_REPORT_PATH = PROJECT_DIR / "reports" / "oof_A1_grouped_v2.json"

FOLDS_PATH = (
    PROJECT_DIR / "data" / "processed" / "training_cv_grouped_v2.json"
)
EVALUATION_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "evaluation_v1"
    / "selection_manifest.json"
)

REPORT_PATH = PROJECT_DIR / "reports" / "oof_A2_grouped_v2.json"
MODEL_PATH = PROJECT_DIR / "models" / "sentiment_A2.joblib"


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    candidates = prepare_training_data()
    evaluation = read_json(EVALUATION_PATH)
    saved_splits = read_json(FOLDS_PATH)
    a1 = read_json(A1_REPORT_PATH)

    held_keys = {
        tuple(row["group_tokens"])
        for row in evaluation["selections"]
    }

    train = [
        row for row in candidates
        if tuple(row["group_key"]) not in held_keys
    ]

    ids = [row["review_id"] for row in train]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training IDs.")

    snapshot = sorted(
        [
            {
                "review_id": row["review_id"],
                "content": row["content"],
                "sentiment": row["sentiment"],
                "group_tokens": list(row["group_key"]),
            }
            for row in train
        ],
        key=lambda row: row["review_id"],
    )

    fingerprint = hashlib.sha256(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if not (
        fingerprint == saved_splits["training_fingerprint"]
        == a1["training_fingerprint"]
    ):
        raise ValueError("Training data differs from A1.")

    if (
        saved_splits["version"] != "grouped_v2"
        or a1["split_version"] != "grouped_v2"
    ):
        raise ValueError("Unexpected split version.")

    keys = sorted({tuple(row["group_key"]) for row in train})
    key_to_id = {key: index for index, key in enumerate(keys)}

    groups = np.array([
        key_to_id[tuple(row["group_key"])]
        for row in train
    ])
    x = [row["content"] for row in train]
    y = np.array([row["sentiment"] for row in train])

    # A1 raporundaki kayıt ve etiketlerle de eşleştir.
    a1_predictions = {
        row["review_id"]: row for row in a1["predictions"]
    }

    if set(ids) != set(a1_predictions):
        raise ValueError("A1 and A2 training IDs differ.")

    for index, row in enumerate(train):
        old = a1_predictions[row["review_id"]]
        if (
            old["actual"] != row["sentiment"]
            or old["group_id"] != int(groups[index])
        ):
            raise ValueError("A1 label or group mismatch.")

    a1_folds = {row["fold"]: row for row in a1["folds"]}
    fold_ids = [row["fold"] for row in saved_splits["folds"]]

    if sorted(fold_ids) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected exactly five saved folds.")

    if set(a1_folds) != set(fold_ids):
        raise ValueError("A1 fold numbers differ.")

    # Yeni bölme üretme; kayıtlı doğrulama ID'lerini kullan.
    splits = []
    coverage = np.zeros(len(train), dtype=int)

    for fold in saved_splits["folds"]:
        number = fold["fold"]
        validation_ids = fold["validation_review_ids"]
        val_set = set(validation_ids)

        if (
            len(validation_ids) != len(val_set)
            or not val_set.issubset(set(ids))
        ):
            raise ValueError(f"Invalid IDs in fold {number}.")

        if val_set != set(a1_folds[number]["validation_review_ids"]):
            raise ValueError(f"A1 fold {number} differs.")

        val_idx = [
            index for index, review_id in enumerate(ids)
            if review_id in val_set
        ]
        fit_idx = [
            index for index, review_id in enumerate(ids)
            if review_id not in val_set
        ]

        if set(groups[fit_idx]) & set(groups[val_idx]):
            raise ValueError(f"Group overlap in fold {number}.")

        if (
            set(y[fit_idx]) != set(LABELS)
            or set(y[val_idx]) != set(LABELS)
        ):
            raise ValueError(f"Missing class in fold {number}.")

        coverage[val_idx] += 1
        splits.append((number, fit_idx, val_idx))

    if not np.all(coverage == 1):
        raise ValueError("Each record must be validated once.")

    template = joblib.load(A1_MODEL_PATH)

    if template.named_steps["classifier"].class_weight is not None:
        raise ValueError("Expected the original unweighted A1 model.")

    print("\nA2 — balanced class weights")
    print("scikit-learn:", sklearn.__version__)
    print("Training records:", len(train))
    print("Training groups:", len(keys))
    print("Data fingerprint matches A1: True")
    print("Saved folds match A1: True")

    oof = np.full(len(train), "", dtype=object)
    fold_reports = []

    for number, fit_idx, val_idx in splits:
        counts = Counter(y[fit_idx])
        class_weights = {
            label: len(fit_idx) / (len(LABELS) * counts[label])
            for label in LABELS
        }

        model = clone(template)
        model.set_params(classifier__class_weight="balanced")

        model.fit(
            [x[index] for index in fit_idx],
            y[fit_idx],
        )
        predictions = model.predict(
            [x[index] for index in val_idx]
        )
        oof[val_idx] = predictions

        weights = group_weights(groups[val_idx])
        record_f1 = macro_f1(y[val_idx], predictions)
        group_f1 = macro_f1(y[val_idx], predictions, weights)
        old_f1 = a1_folds[number]["group_macro_f1"]

        print(f"\nFold {number}")
        print(
            "Training class weights:",
            {label: round(value, 4)
             for label, value in class_weights.items()},
        )
        print(f"A1 group macro-F1: {old_f1:.4f}")
        print(f"A2 group macro-F1: {group_f1:.4f}")
        print(f"Difference: {group_f1 - old_f1:+.4f}")

        fold_reports.append({
            "fold": number,
            "class_weights": class_weights,
            "record_macro_f1": record_f1,
            "group_macro_f1": group_f1,
            "a1_group_macro_f1": old_f1,
            "validation_review_ids": [ids[i] for i in val_idx],
            "group_classification_report": classification_report(
                y[val_idx],
                predictions,
                labels=LABELS,
                sample_weight=weights,
                output_dict=True,
                zero_division=0,
            ),
        })

    if np.any(oof == ""):
        raise ValueError("Missing OOF predictions.")

    # Aynı özelliklere sahip grup üyelerinde tutarlılığı kontrol et.
    group_results = {}
    for group, actual, predicted in zip(groups, y, oof):
        result = (str(actual), str(predicted))
        if group_results.setdefault(int(group), result) != result:
            raise ValueError("Inconsistent results within a group.")

    weights = group_weights(groups)

    record_report = print_report(
        "A2 — pooled OOF, all records", y, oof
    )
    group_report = print_report(
        "A2 — pooled OOF, equal group weights", y, oof, weights
    )

    pooled = {
        "record_macro_f1": macro_f1(y, oof),
        "group_macro_f1": macro_f1(y, oof, weights),
    }
    fold_means = {
        metric: float(np.mean([f[metric] for f in fold_reports]))
        for metric in pooled
    }

    a1_group = a1["group_classification_report"]
    improvement = (
        pooled["group_macro_f1"]
        - a1["pooled_oof"]["group_macro_f1"]
    )
    positive_recall_drop = (
        a1_group["positive"]["recall"]
        - group_report["positive"]["recall"]
    )
    improved_folds = sum(
        fold["group_macro_f1"] > fold["a1_group_macro_f1"]
        for fold in fold_reports
    )

    # Önceden belirlediğimiz dört koşul.
    checks = {
        "group_macro_f1_gain_at_least_0.02": improvement >= 0.02,
        "group_neutral_f1_increased": (
            group_report["neutral"]["f1-score"]
            > a1_group["neutral"]["f1-score"]
        ),
        "group_positive_recall_drop_at_most_0.05": (
            positive_recall_drop <= 0.05
        ),
        "improved_in_at_least_4_folds": improved_folds >= 4,
    }

    print("\nPooled OOF comparison:")
    for metric, value in pooled.items():
        print(
            f"{metric}: "
            f"A1={a1['pooled_oof'][metric]:.4f} | A2={value:.4f}"
        )

    print("\nA2 mean of fold scores:", fold_means)
    print("Group macro-F1 gain:", round(improvement, 6))
    print("Group positive recall drop:", round(positive_recall_drop, 6))
    print("Improved folds:", improved_folds)

    print("\nPredefined selection criteria:")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    preferred = "A2" if all(checks.values()) else "A1"
    print("Preferred by predefined criteria:", preferred)

    # Karşılaştırma sonrası A2'nin bütün eğitim havuzuyla
    # eğitilmiş sürümünü ayrı kaydet.
    print("\nTraining A2 on all training records...")
    final_model = clone(template)
    final_model.set_params(classifier__class_weight="balanced")
    final_model.fit(x, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    report = {
        "experiment": "A2_OOF_grouped_v2",
        "sklearn_version": sklearn.__version__,
        "class_weight": "balanced",
        "split_version": "grouped_v2",
        "training_fingerprint": fingerprint,
        "training_records": len(train),
        "training_groups": len(keys),
        "labels": LABELS,
        "folds": fold_reports,
        "fold_means": fold_means,
        "pooled_oof": pooled,
        "record_classification_report": record_report,
        "group_classification_report": group_report,
        "prediction_counts": {
            label: int(np.count_nonzero(oof == label))
            for label in LABELS
        },
        "selection_checks": checks,
        "preferred_by_predefined_criteria": preferred,
        "predictions": [
            {
                "review_id": ids[index],
                "group_id": int(groups[index]),
                "actual": str(y[index]),
                "predicted": str(oof[index]),
            }
            for index in range(len(train))
        ],
        "final_test_scored": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nOOF prediction counts:", report["prediction_counts"])
    print("Saved model:", MODEL_PATH)
    print("Saved report:", REPORT_PATH)
    print("Final test predictions: not generated.")


if __name__ == "__main__":
    main()