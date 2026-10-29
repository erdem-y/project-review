from collections import Counter
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

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

REPORT_PATH = (
    PROJECT_DIR / "reports" / "oof_A1_grouped_v2.json"
)

FOLDS_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "training_cv_grouped_v2.json"
)


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

    train_keys = {
        tuple(row["group_key"])
        for row in train
    }

    if train_keys & held_keys:
        raise ValueError(
            "Evaluation/training group overlap detected."
        )

    review_ids = [row["review_id"] for row in train]

    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Duplicate training review IDs.")

    expected_counts = {
        "negative": 462,
        "neutral": 104,
        "positive": 1954,
    }
    actual_counts = Counter(
        row["sentiment"] for row in train
    )

    if (
        len(train) != 2520
        or dict(actual_counts) != expected_counts
        or len(train_keys) != 1335
    ):
        raise ValueError(
            "Training data differs from the expected A1 dataset: "
            f"{len(train)} records, "
            f"{len(train_keys)} groups, "
            f"{dict(actual_counts)}"
        )

    x = np.array(
        [row["content"] for row in train],
        dtype=object,
    )
    y = np.array(
        [row["sentiment"] for row in train],
        dtype=object,
    )

    key_to_id = {
        key: index
        for index, key in enumerate(sorted(train_keys))
    }
    groups = np.array(
        [
            key_to_id[tuple(row["group_key"])]
            for row in train
        ]
    )

    # Kayıtlı A1'in ayarlarını kullan.
    # clone(), öğrenilmiş sözlüğü ve katsayıları taşımaz.
    template = joblib.load(MODEL_PATH)

    if template.named_steps["classifier"].class_weight is not None:
        raise ValueError("Expected A1 with class_weight=None.")

    # Her metin grubu tek bir duygu etiketine sahip olmalı.
    group_labels = {}

    for group_id, label in zip(groups, y):
        group_id = int(group_id)
        label = str(label)

        if (
            group_id in group_labels
            and group_labels[group_id] != label
        ):
            raise ValueError(
                f"Conflicting labels in group {group_id}."
            )

        group_labels[group_id] = label

    unique_groups = np.array(sorted(group_labels))
    unique_labels = np.array(
        [
            group_labels[int(group)]
            for group in unique_groups
        ]
    )

    if min(Counter(unique_labels).values()) < 5:
        raise ValueError(
            "Each class needs at least five groups."
        )

    # Bölmeyi kayıtlar üzerinde değil, tekil gruplar üzerinde yap.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    splits = []
    fold_manifest = []
    planned_coverage = np.zeros(len(train), dtype=int)

    print("\nGroup-stratified validation folds:")

    for fold, (fit_group_idx, val_group_idx) in enumerate(
        splitter.split(unique_groups, unique_labels),
        start=1,
    ):
        fit_groups = unique_groups[fit_group_idx]
        val_groups = unique_groups[val_group_idx]

        # Seçilen grupların bütün kayıtlarını aynı bölümde tut.
        fit_idx = np.flatnonzero(
            np.isin(groups, fit_groups)
        )
        val_idx = np.flatnonzero(
            np.isin(groups, val_groups)
        )

        if set(groups[fit_idx]) & set(groups[val_idx]):
            raise ValueError(
                f"Group overlap in fold {fold}."
            )

        if (
            set(y[fit_idx]) != set(LABELS)
            or set(y[val_idx]) != set(LABELS)
        ):
            raise ValueError(
                f"Missing class in fold {fold}."
            )

        splits.append((fit_idx, val_idx))
        planned_coverage[val_idx] += 1

        group_counts = {
            label: int(
                np.count_nonzero(
                    unique_labels[val_group_idx] == label
                )
            )
            for label in LABELS
        }

        record_counts = {
            label: int(np.count_nonzero(y[val_idx] == label))
            for label in LABELS
        }

        print(
            f"Fold {fold}: "
            f"groups={group_counts} | "
            f"records={record_counts}"
        )

        fold_manifest.append({
            "fold": fold,
            "validation_review_ids": sorted(
                train[index]["review_id"]
                for index in val_idx
            ),
            "validation_group_counts": group_counts,
            "validation_record_counts": record_counts,
        })

    if not np.all(planned_coverage == 1):
        raise ValueError(
            "Each training record must appear in "
            "exactly one validation fold."
        )

    # A2'de aynı veri ve gruplamanın kullanıldığını
    # doğrulamak için eğitim havuzunun parmak izini oluştur.
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

    split_document = {
        "version": "grouped_v2",
        "method": "StratifiedKFold over unique text groups",
        "random_state": 42,
        "n_splits": 5,
        "training_fingerprint": fingerprint,
        "training_records": len(train),
        "training_groups": len(unique_groups),
        "folds": fold_manifest,
    }

    # Sonuçları görmeden fold'ları kaydet.
    # Önceden kaydedilen farklı bir bölmenin üzerine yazma.
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if FOLDS_PATH.exists():
        with FOLDS_PATH.open(encoding="utf-8") as file:
            saved_splits = json.load(file)

        if saved_splits != split_document:
            raise ValueError(
                "Training data or folds differ "
                "from the saved v2 split."
            )
    else:
        with FOLDS_PATH.open("x", encoding="utf-8") as file:
            json.dump(
                split_document,
                file,
                ensure_ascii=False,
                indent=2,
            )

    print("\nFixed CV splits:", FOLDS_PATH)

    oof = np.full(len(train), "", dtype=object)
    majority_oof = np.full(len(train), "", dtype=object)
    coverage = np.zeros(len(train), dtype=int)
    fold_reports = []

    print("\nOOF diagnostics — A1, grouped v2")
    print("Training records:", len(train))
    print("Training groups:", len(train_keys))

    for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
        model = clone(template)

        model.fit(
            x[fit_idx].tolist(),
            y[fit_idx].tolist(),
        )
        predicted = model.predict(x[val_idx].tolist())

        # Baseline yalnızca fold'un eğitim etiketlerini kullanır.
        majority = Counter(
            y[fit_idx]
        ).most_common(1)[0][0]

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
            "record_macro_f1": macro_f1(
                y[val_idx], predicted
            ),
            "group_macro_f1": macro_f1(
                y[val_idx], predicted, weights
            ),
            "majority_record_macro_f1": macro_f1(
                y[val_idx], majority_predictions
            ),
            "majority_group_macro_f1": macro_f1(
                y[val_idx], majority_predictions, weights
            ),
            "majority_class": str(majority),
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
            f"\n  A1 records: "
            f"{metrics['record_macro_f1']:.4f}"
            f" | groups: {metrics['group_macro_f1']:.4f}"
            f"\n  Majority records: "
            f"{metrics['majority_record_macro_f1']:.4f}"
            f" | groups: "
            f"{metrics['majority_group_macro_f1']:.4f}"
        )

    if not np.all(coverage == 1):
        raise ValueError(
            "Every record must receive one OOF prediction."
        )

    # Aynı grubun etiketi ve tahmini tutarlı olmalı.
    group_results = {}

    for group, actual, predicted in zip(groups, y, oof):
        result = (str(actual), str(predicted))
        previous = group_results.setdefault(
            int(group), result
        )

        if previous != result:
            raise ValueError(
                "Inconsistent labels or predictions "
                f"in group {group}."
            )

    weights = group_weights(groups)

    record_report = print_report(
        "A1 — pooled OOF, all records",
        y,
        oof,
    )

    group_report = print_report(
        "A1 — pooled OOF, equal group weights",
        y,
        oof,
        weights,
    )

    pooled = {
        "record_macro_f1": macro_f1(y, oof),
        "group_macro_f1": macro_f1(y, oof, weights),
        "majority_record_macro_f1": macro_f1(
            y, majority_oof
        ),
        "majority_group_macro_f1": macro_f1(
            y, majority_oof, weights
        ),
    }

    fold_means = {
        metric: float(
            np.mean([
                row[metric]
                for row in fold_reports
            ])
        )
        for metric in pooled
    }

    print("\nMean of fold scores:")
    for name, value in fold_means.items():
        print(f"{name}: {value:.4f}")

    print("\nPooled OOF scores:")
    for name, value in pooled.items():
        print(f"{name}: {value:.4f}")

    prediction_counts = {
        label: int(np.count_nonzero(oof == label))
        for label in LABELS
    }
    print("\nOOF prediction counts:", prediction_counts)

    report = {
        "experiment": "A1_OOF_grouped_v2",
        "split_version": "grouped_v2",
        "training_fingerprint": fingerprint,
        "training_records": len(train),
        "training_groups": len(train_keys),
        "labels": LABELS,
        "folds": fold_reports,
        "fold_means": fold_means,
        "pooled_oof": pooled,
        "record_classification_report": record_report,
        "group_classification_report": group_report,
        "prediction_counts": prediction_counts,
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
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved report:", REPORT_PATH)
    print("Final test predictions: not generated.")


if __name__ == "__main__":
    main()