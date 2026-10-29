from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
import csv
import json

import joblib
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from prepare_training_data import prepare_training_data
from text_features import tokenize_review


LABELS = ["negative", "neutral", "positive"]
SEED = 42


def macro_f1(actual, predicted):
    return float(f1_score(
        actual,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    ))


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=tokenize_review,
                token_pattern=None,
                lowercase=False,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight=None,
                max_iter=2000,
            ),
        ),
    ])


def load_data(project_dir):
    candidates = prepare_training_data()

    folder = (
        project_dir / "data" / "processed" / "evaluation_v1"
    )

    with (folder / "selection_manifest.json").open(
        encoding="utf-8-sig"
    ) as file:
        manifest = json.load(file)

    with (folder / "labelingyeni.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter=";")

        if reader.fieldnames != [
            "review_id",
            "content",
            "sentiment",
            "notes",
        ]:
            raise ValueError("Unexpected labeling columns.")

        gold_rows = list(reader)

    selections = manifest["selections"]
    gold = {row["review_id"]: row for row in gold_rows}
    selected_ids = {
        item["review_id"] for item in selections
    }

    if not (
        len(selections)
        == len(selected_ids)
        == len(gold_rows)
        == len(gold)
        == 130
    ):
        raise ValueError(
            "Expected 130 unique evaluation records."
        )

    if set(gold) != selected_ids:
        raise ValueError(
            "Labeling IDs do not match the manifest."
        )

    if manifest["candidate_count"] != len(candidates):
        raise ValueError(
            "Candidate count changed after selection."
        )

    candidate_ids = {
        row["review_id"] for row in candidates
    }

    if len(candidate_ids) != len(candidates):
        raise ValueError("Duplicate candidate IDs.")

    for row in gold_rows:
        if None in row or any(
            value is None for value in row.values()
        ):
            raise ValueError("Malformed labeling CSV row.")

        if row["sentiment"] not in LABELS + ["unclear"]:
            raise ValueError(
                f"Invalid label: {row['review_id']}"
            )

    split_counts = Counter(
        item["split"] for item in selections
    )

    if split_counts != Counter({"dev": 30, "test": 100}):
        raise ValueError("Unexpected dev/test allocation.")

    current_groups = defaultdict(set)

    for row in candidates:
        current_groups[row["group_key"]].add(
            row["review_id"]
        )

    held_keys = set()

    for item in selections:
        key = tuple(item["group_tokens"])

        if key in held_keys:
            raise ValueError(
                "Repeated evaluation group."
            )

        held_keys.add(key)
        expected_ids = set(item["group_review_ids"])

        if item["review_id"] not in expected_ids:
            raise ValueError(
                "Selected ID missing from its group."
            )

        if current_groups.get(key) != expected_ids:
            raise ValueError(
                "Candidate groups changed after selection."
            )

        original_text = gold[item["review_id"]]["content"]

        if tuple(tokenize_review(original_text)) != key:
            raise ValueError(
                f"Text mismatch: {item['review_id']}"
            )

    # Belirsiz etiketli değerlendirme kayıtları dahil,
    # bütün değerlendirme gruplarını eğitimden çıkar.
    train = [
        row for row in candidates
        if row["group_key"] not in held_keys
    ]

    overlap = {
        row["group_key"] for row in train
    } & held_keys

    train_ids = {
        row["review_id"] for row in train
    }

    if overlap or (train_ids & selected_ids):
        raise ValueError(
            "Evaluation data leaked into training."
        )

    dev_all = [
        gold[item["review_id"]]
        for item in selections
        if item["split"] == "dev"
    ]

    dev = [
        row for row in dev_all
        if row["sentiment"] != "unclear"
    ]

    if not train or not dev:
        raise ValueError(
            "Empty training/development data."
        )

    print("\nTrain records:", len(train))
    print("Usable dev records:", len(dev))
    print("Dev unclear records:", len(dev_all) - len(dev))
    print(
        "Evaluation/training group overlap:",
        len(overlap),
    )
    print(
        "Training labels:",
        dict(Counter(row["sentiment"] for row in train)),
    )

    return train, dev


def run_cv(model, train):
    x = [row["content"] for row in train]
    y = [row["sentiment"] for row in train]

    keys = sorted({
        row["group_key"] for row in train
    })

    group_ids = {
        key: index
        for index, key in enumerate(keys)
    }

    groups = [
        group_ids[row["group_key"]]
        for row in train
    ]

    group_labels = defaultdict(set)

    for row in train:
        group_labels[row["group_key"]].add(
            row["sentiment"]
        )

    if any(
        len(labels) != 1
        for labels in group_labels.values()
    ):
        raise ValueError(
            "Conflicting training group labels."
        )

    counts = Counter(
        next(iter(labels))
        for labels in group_labels.values()
    )

    if any(counts[label] < 5 for label in LABELS):
        raise ValueError(
            "Need at least five groups per class for 5-fold CV."
        )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    folds = []

    for number, (fit_idx, valid_idx) in enumerate(
        cv.split(x, y, groups),
        start=1,
    ):
        overlap = {
            groups[i] for i in fit_idx
        } & {
            groups[i] for i in valid_idx
        }

        if overlap:
            raise ValueError("CV group leakage.")

        fit_x = [x[i] for i in fit_idx]
        fit_y = [y[i] for i in fit_idx]
        valid_x = [x[i] for i in valid_idx]
        valid_y = [y[i] for i in valid_idx]

        if (
            set(fit_y) != set(LABELS)
            or set(valid_y) != set(LABELS)
        ):
            raise ValueError(
                "A CV fold is missing a class; review the split."
            )

        # Her bölümde TF-IDF dahil yeni bir pipeline öğrenir.
        estimator = clone(model)
        estimator.fit(fit_x, fit_y)
        predicted = estimator.predict(valid_x)

        majority = Counter(fit_y).most_common(1)[0][0]

        result = {
            "fold": number,
            "train_records": len(fit_idx),
            "validation_records": len(valid_idx),
            "validation_labels": dict(Counter(valid_y)),
            "group_overlap": len(overlap),
            "macro_f1": macro_f1(valid_y, predicted),
            "accuracy": float(
                accuracy_score(valid_y, predicted)
            ),
            "majority_macro_f1": macro_f1(
                valid_y,
                [majority] * len(valid_y),
            ),
            "validation_review_ids": [
                train[i]["review_id"]
                for i in valid_idx
            ],
        }

        folds.append(result)

        print(
            f"Fold {number}: "
            f"macro-F1={result['macro_f1']:.4f}"
        )

    scores = [
        fold["macro_f1"] for fold in folds
    ]

    return {
        "label_source": (
            "star-aligned training labels, "
            "not manual gold labels"
        ),
        "seed": SEED,
        "folds": folds,
        "macro_f1_mean": mean(scores),
        "macro_f1_std": pstdev(scores),
        "majority_macro_f1_mean": mean(
            fold["majority_macro_f1"]
            for fold in folds
        ),
    }


def main():
    project_dir = Path(__file__).resolve().parent

    print("scikit-learn:", sklearn.__version__)
    train, dev = load_data(project_dir)
    model = build_model()

    print("\nGrouped 5-fold CV — training labels:")
    cv_result = run_cv(model, train)

    print(
        "CV mean macro-F1:",
        f"{cv_result['macro_f1_mean']:.4f}",
    )
    print(
        "CV standard deviation:",
        f"{cv_result['macro_f1_std']:.4f}",
    )

    x_train = [row["content"] for row in train]
    y_train = [row["sentiment"] for row in train]
    x_dev = [row["content"] for row in dev]
    y_dev = [row["sentiment"] for row in dev]

    print("\nTraining final A1 on all training records...")
    model.fit(x_train, y_train)
    predicted = model.predict(x_dev)

    majority = Counter(y_train).most_common(1)[0][0]

    dev_result = {
        "label_source": "manual labels",
        "records": len(dev),
        "accuracy": float(
            accuracy_score(y_dev, predicted)
        ),
        "macro_f1": macro_f1(y_dev, predicted),
        "majority_macro_f1": macro_f1(
            y_dev,
            [majority] * len(y_dev),
        ),
        "classification_report": classification_report(
            y_dev,
            predicted,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_dev,
            predicted,
            labels=LABELS,
        ).tolist(),
        "predictions": [
            {
                "review_id": row["review_id"],
                "text": row["content"],
                "actual": actual,
                "predicted": str(prediction),
            }
            for row, actual, prediction in zip(
                dev, y_dev, predicted
            )
        ],
    }

    print("\nDevelopment results — manual labels:")
    print(
        "Majority macro-F1:",
        f"{dev_result['majority_macro_f1']:.4f}",
    )
    print(
        "A1 accuracy:",
        f"{dev_result['accuracy']:.4f}",
    )
    print(
        "A1 macro-F1:",
        f"{dev_result['macro_f1']:.4f}",
    )

    print(classification_report(
        y_dev,
        predicted,
        labels=LABELS,
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix: rows=actual, columns=predicted")
    print("Class order:", LABELS)
    print(confusion_matrix(
        y_dev,
        predicted,
        labels=LABELS,
    ))

    print("\nIncorrect dev predictions:")

    for item in dev_result["predictions"]:
        if item["actual"] != item["predicted"]:
            print(f"\nID: {item['review_id']}")
            print(
                f"Actual: {item['actual']} | "
                f"Predicted: {item['predicted']}"
            )
            print(item["text"])

    names = (
        model.named_steps["tfidf"]
        .get_feature_names_out()
    )
    classifier = model.named_steps["classifier"]
    important_features = {}

    for index, label in enumerate(classifier.classes_):
        weights = classifier.coef_[index]
        top = weights.argsort()[-15:][::-1]

        items = [
            {
                "feature": str(names[j]),
                "coefficient": float(weights[j]),
            }
            for j in top
        ]

        important_features[str(label)] = items

        print(f"\nHighest coefficients for {label}:")

        for item in items:
            print(
                f"{item['feature']!r}: "
                f"{item['coefficient']:.4f}"
            )

    report = {
        "experiment": "A1",
        "sklearn_version": sklearn.__version__,
        "parameters": {
            "ngram_range": [1, 2],
            "C": 1.0,
            "class_weight": None,
            "max_iter": 2000,
        },
        "train_records": len(train),
        "train_labels": dict(Counter(y_train)),
        "train_review_ids": [
            row["review_id"] for row in train
        ],
        "label_order": LABELS,
        "cv": cv_result,
        "dev": dev_result,
        "vocabulary_size": len(names),
        "highest_coefficients": important_features,
        "final_test_scored": False,
    }

    model_dir = project_dir / "models"
    report_dir = project_dir / "reports"

    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    model_path = model_dir / "sentiment_A1.joblib"
    report_path = report_dir / "baseline_A1_dev.json"

    joblib.dump(model, model_path)

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\nSaved model:", model_path)
    print("Saved report:", report_path)
    print("Final test predictions: not generated.")


if __name__ == "__main__":
    main()