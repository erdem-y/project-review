from collections import Counter
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from prepare_training_data import prepare_training_data


PROJECT_DIR = Path(__file__).resolve().parent
LABELS = ["negative", "neutral", "positive"]

MODEL_PATH = PROJECT_DIR / "models" / "sentiment_A1.joblib"
MANIFEST_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "evaluation_v1"
    / "selection_manifest.json"
)
REPORT_PATH = PROJECT_DIR / "reports" / "oof_A1.json"


def macro_f1(actual, predicted, weights=None):
    return float(
        f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="macro",
            sample_weight=weights,
            zero_division=0,
        )
    )


def group_weights(group_ids):
    counts = Counter(group_ids)
    return np.array(
        [1.0 / counts[group] for group in group_ids],
        dtype=float,
    )


def print_report(title, actual, predicted, weights=None):
    print(f"\n{title}")
    print(
        classification_report(
            actual,
            predicted,
            labels=LABELS,
            sample_weight=weights,
            digits=4,
            zero_division=0,
        )
    )

    return classification_report(
        actual,
        predicted,
        labels=LABELS,
        sample_weight=weights,
        output_dict=True,
        zero_division=0,
    )


def main():
    candidates = prepare_training_data()

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    held_keys = {
        tuple(selection["group_tokens"])
        for selection in manifest["selections"]
    }

    train = [
        row
        for row in candidates
        if tuple(row["group_key"]) not in held_keys
    ]

    train_keys = {tuple(row["group_key"]) for row in train}

    if train_keys & held_keys:
        raise ValueError("Evaluation/training group overlap detected.")

    # Beklenen A1 eğitim havuzunun korunduğunu kontrol et.
    expected_counts = {
        "negative": 462,
        "neutral": 104,
        "positive": 1954,
    }
    actual_counts = Counter(row["sentiment"] for row in train)

    if len(train) != 2520 or dict(actual_counts) != expected_counts:
        raise ValueError(
            "Training data differs from the original A1 run: "
            f"{len(train)} records, {dict(actual_counts)}"
        )

    x = np.array([row["content"] for row in train], dtype=object)
    y = np.array([row["sentiment"] for row in train], dtype=object)

    key_to_id = {
        key: index
        for index, key in enumerate(sorted(train_keys))
    }
    groups = np.array(
        [key_to_id[tuple(row["group_key"])] for row in train]
    )

    # Kayıtlı A1'in ayarlarını kullan.
    # clone(), öğrenilmiş sözlüğü ve katsayıları taşımaz.
    template = joblib.load(MODEL_PATH)

    if template.named_steps["classifier"].class_weight is not None:
        raise ValueError("Expected A1 with class_weight=None.")

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof = np.full(len(train), "", dtype=object)
    majority_oof = np.full(len(train), "", dtype=object)
    coverage = np.zeros(len(train), dtype=int)
    fold_reports = []

    print("\nOOF diagnostics — A1")
    print("Training records:", len(train))
    print("Training groups:", len(train_keys))

    for fold, (fit_idx, val_idx) in enumerate(
        cv.split(x, y, groups),
        start=1,
    ):
        if set(groups[fit_idx]) & set(groups[val_idx]):
            raise ValueError(f"Group overlap in fold {fold}.")

        if set(y[fit_idx]) != set(LABELS):
            raise ValueError(f"Missing training class in fold {fold}.")

        model = clone(template)
        model.fit(x[fit_idx].tolist(), y[fit_idx].tolist())

        predicted = model.predict(x[val_idx].tolist())

        # Baseline yalnızca fold'un eğitim etiketlerini kullanır.
        majority = Counter(y[fit_idx]).most_common(1)[0][0]
        majority_predictions = np.full(
            len(val_idx),
            majority,
            dtype=object,
        )

        oof[val_idx] = predicted
        majority_oof[val_idx] = majority_predictions
        coverage[val_idx] += 1

        weights = group_weights(groups[val_idx])

        metrics = {
            "fold": fold,
            "record_macro_f1": macro_f1(y[val_idx], predicted),
            "group_macro_f1": macro_f1(
                y[val_idx], predicted, weights
            ),
            "majority_record_macro_f1": macro_f1(
                y[val_idx], majority_predictions
            ),
            "majority_group_macro_f1": macro_f1(
                y[val_idx], majority_predictions, weights
            ),
            "validation_review_ids": [
                train[index]["review_id"]
                for index in val_idx
            ],
            "record_classification_report": classification_report(
                y[val_idx],
                predicted,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            ),
            "group_classification_report": classification_report(
                y[val_idx],
                predicted,
                labels=LABELS,
                sample_weight=weights,
                output_dict=True,
                zero_division=0,
            ),
        }
        fold_reports.append(metrics)

        print(
            f"\nFold {fold}:"
            f"\n  A1 records: {metrics['record_macro_f1']:.4f}"
            f" | groups: {metrics['group_macro_f1']:.4f}"
            f"\n  Majority records: "
            f"{metrics['majority_record_macro_f1']:.4f}"
            f" | groups: {metrics['majority_group_macro_f1']:.4f}"
        )

    if not np.all(coverage == 1):
        raise ValueError("Every record must receive one OOF prediction.")

    # Aynı grubun etiketi ve tahmini tutarlı olmalı.
    group_results = {}
    for group, actual, predicted in zip(groups, y, oof):
        result = (actual, predicted)
        previous = group_results.setdefault(int(group), result)

        if previous != result:
            raise ValueError(
                f"Inconsistent labels or predictions in group {group}."
            )

    weights = group_weights(groups)

    record_report = print_report(
        "A1 — pooled OOF, all records", y, oof
    )
    group_report = print_report(
        "A1 — pooled OOF, equal group weights", y, oof, weights
    )

    pooled = {
        "record_macro_f1": macro_f1(y, oof),
        "group_macro_f1": macro_f1(y, oof, weights),
        "majority_record_macro_f1": macro_f1(y, majority_oof),
        "majority_group_macro_f1": macro_f1(
            y, majority_oof, weights
        ),
    }

    fold_means = {
        metric: float(np.mean([row[metric] for row in fold_reports]))
        for metric in pooled
    }

    print("\nMean of fold scores:")
    for name, value in fold_means.items():
        print(f"{name}: {value:.4f}")

    print("\nPooled OOF scores:")
    for name, value in pooled.items():
        print(f"{name}: {value:.4f}")

    # Önceki A1 çıktısının yeniden üretildiğini kontrol et.
    reproduced = (
        round(fold_means["record_macro_f1"], 4) == 0.5366
    )
    print("\nOriginal A1 CV mean reproduced:", reproduced)

    if not reproduced:
        print(
            "STOP before A2: investigate the difference "
            "from the original A1 result."
        )

    report = {
        "experiment": "A1_OOF",
        "original_cv_mean_reproduced_to_4_decimals": reproduced,
        "training_records": len(train),
        "training_groups": len(train_keys),
        "labels": LABELS,
        "folds": fold_reports,
        "fold_means": fold_means,
        "pooled_oof": pooled,
        "record_classification_report": record_report,
        "group_classification_report": group_report,
        "predictions": [
            {
                "review_id": row["review_id"],
                "group_id": int(groups[index]),
                "actual": str(y[index]),
                "predicted": str(oof[index]),
                "majority_predicted": str(majority_oof[index]),
            }
            for index, row in enumerate(train)
        ],
        "final_test_scored": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nSaved report:", REPORT_PATH)


if __name__ == "__main__":
    main()