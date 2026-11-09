from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import json

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEEDS = (42, 43, 44)
LOWER = 0.50
UPPER = 0.65

TARGETS = {
    "coverage": 0.75,
    "accuracy": 0.88,
    "negative_coverage": 0.70,
}
TOLERANCE = 0.01


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def evaluate(actual, probability):
    decisions = np.where(
        probability < LOWER,
        "negative",
        np.where(
            probability >= UPPER,
            "positive",
            "needs_review",
        ),
    )

    accepted = decisions != "needs_review"
    negative = actual == "negative"
    positive = actual == "positive"

    counts = {
        "negative_correct": int(
            np.sum(negative & (decisions == "negative"))
        ),
        "negative_as_positive": int(
            np.sum(negative & (decisions == "positive"))
        ),
        "negative_review": int(
            np.sum(negative & ~accepted)
        ),
        "positive_correct": int(
            np.sum(positive & (decisions == "positive"))
        ),
        "positive_as_negative": int(
            np.sum(positive & (decisions == "negative"))
        ),
        "positive_review": int(
            np.sum(positive & ~accepted)
        ),
    }

    metrics = {
        "coverage": float(accepted.mean()),
        "accuracy": (
            float(np.mean(decisions[accepted] == actual[accepted]))
            if accepted.any()
            else 0.0
        ),
        "negative_coverage": float(accepted[negative].mean()),
        "needs_review_rate": float((~accepted).mean()),
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
            else 0.0
        ),
    }

    for label in ("negative", "positive"):
        selected = decisions == label
        metrics[label + "_precision"] = (
            float(np.mean(actual[selected] == label))
            if selected.any()
            else None
        )

    checks = {}

    for name, target in TARGETS.items():
        value = metrics[name]

        # Küçük sayısal yuvarlama farklarını dikkate al.
        if value + 1e-12 >= target:
            checks[name] = "PASS"
        elif value + 1e-12 >= target - TOLERANCE:
            checks[name] = "TOLERANCE"
        else:
            checks[name] = "FAIL"

    return metrics, counts, checks, decisions


def main():
    # 1. Aynı B1 eğitim verisini kullandığımızı doğrula.
    csv_path = DATA / "training_B1_v3.csv"
    fingerprint = sha256(csv_path.read_bytes()).hexdigest()

    old = json.loads(
        (REPORTS / "oof_B1_grouped_v3.json").read_text(
            encoding="utf-8-sig"
        )
    )

    if fingerprint != old["training_csv_sha256"]:
        raise ValueError(
            "Training CSV differs from the original B1 data."
        )

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))

    ids = [row["review_id"] for row in rows]
    x = np.array([row["content"] for row in rows], dtype=object)
    y = np.array([row["sentiment"] for row in rows])
    groups = np.array([int(row["group_id"]) for row in rows])

    if len(set(ids)) != len(ids) or Counter(y) != {
        "negative": 530,
        "positive": 1947,
    }:
        raise ValueError("Unexpected B1 records or labels.")

    # 2. Dev/test kayıtlarının eğitimde olmadığını kontrol et.
    manifest = json.loads(
        (
            DATA / "evaluation_v1" / "selection_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )

    held_ids = set()

    for selection in manifest["selections"]:
        held_ids.add(selection["review_id"])
        held_ids.update(selection["group_review_ids"])

    if set(ids) & held_ids:
        raise ValueError(
            "Development/test records found in training."
        )

    # 3. Grupları hazırla. Her grup değerlendirmede bir oy alır.
    members = {
        int(group): np.flatnonzero(groups == group)
        for group in sorted(set(groups))
    }
    unique_groups = np.array(list(members))
    first = np.array(
        [indices[0] for indices in members.values()]
    )
    group_y = y[first]

    if Counter(group_y) != {
        "negative": 518,
        "positive": 780,
    }:
        raise ValueError("Unexpected group counts.")

    for indices in members.values():
        if len(set(y[indices])) != 1:
            raise ValueError("Conflicting group labels.")

    reference = {
        row["review_id"]: row
        for row in old["predictions"]
    }

    if (
        len(reference) != len(old["predictions"])
        or set(reference) != set(ids)
    ):
        raise ValueError(
            "Original B1 report has different review IDs."
        )

    for row in rows:
        previous = reference[row["review_id"]]

        if (
            previous["actual"] != row["sentiment"]
            or previous["group_id"] != int(row["group_id"])
        ):
            raise ValueError(
                "Original B1 labels or groups differ."
            )

    # 4. B1'i oluştururken kullandığımız ayarları yükle.
    # clone() öğrenilmiş sözlüğü ve katsayıları taşımaz.
    model_path = ROOT / "models" / "sentiment_A1.joblib"
    template = joblib.load(model_path)

    if template.named_steps["classifier"].class_weight is not None:
        raise ValueError(
            "Expected A1 settings without class weighting."
        )

    # 5. Kuralları yeni sonuçlar oluşmadan kaydet.
    plan = {
        "seeds": list(SEEDS),
        "lower": LOWER,
        "upper": UPPER,
        "targets": TARGETS,
        "tolerance_percentage_points": 1,
        "training_csv_sha256": fingerprint,
        "template_sha256": sha256(
            model_path.read_bytes()
        ).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "unit": "one vote per text group",
        "final_test_scored": False,
    }

    plan_path = DATA / "B1_stability_plan.json"

    if plan_path.exists():
        saved_plan = json.loads(
            plan_path.read_text(encoding="utf-8")
        )
        if saved_plan != plan:
            raise ValueError(
                "Saved stability plan differs. "
                "Investigate before rerunning."
            )
    else:
        save_json(plan_path, plan)

    print("Plan saved/verified:", plan_path)
    print(
        f"Training: {len(rows)} records, "
        f"{len(unique_groups)} groups"
    )
    print("Fixed band:", LOWER, UPPER)

    results = []

    # 6. Aynı yöntemle üç farklı grup bölünmesini değerlendir.
    for seed in SEEDS:
        probability = np.full(len(rows), np.nan)
        coverage = np.zeros(len(rows), dtype=int)
        folds = []

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        for fold, (fit_g, val_g) in enumerate(
            cv.split(unique_groups, group_y),
            start=1,
        ):
            fit = np.isin(groups, unique_groups[fit_g])
            val = np.isin(groups, unique_groups[val_g])

            val_ids = [
                ids[index]
                for index in np.flatnonzero(val)
            ]

            if seed == 42:
                original_ids = old["folds"][fold - 1][
                    "validation_review_ids"
                ]
                if set(val_ids) != set(original_ids):
                    raise ValueError(
                        "Seed 42 folds differ from original B1."
                    )

            model = clone(template)
            model.fit(x[fit], y[fit])

            positive_column = list(model.classes_).index(
                "positive"
            )
            probability[val] = model.predict_proba(
                x[val]
            )[:, positive_column]

            coverage[val] += 1

            folds.append({
                "fold": fold,
                "validation_review_ids": val_ids,
            })

            print(
                f"Seed {seed}, fold {fold}/5 completed",
                flush=True,
            )

        if (
            not np.all(coverage == 1)
            or not np.all(np.isfinite(probability))
        ):
            raise ValueError(
                "Invalid OOF coverage or probabilities."
            )

        for indices in members.values():
            if not np.allclose(
                probability[indices],
                probability[indices[0]],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError(
                    "Predictions differ within a text group."
                )

        # Seed 42 eski B1 sonucunu yeniden üretmeli.
        if seed == 42:
            original = np.array([
                reference[review_id]["positive_probability"]
                for review_id in ids
            ])

            if not np.allclose(
                probability,
                original,
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(
                    "Seed 42 probabilities differ from "
                    "original B1. Stop and investigate."
                )

            print("Original B1 probabilities reproduced: True")

        metrics, counts, checks, decisions = evaluate(
            group_y,
            probability[first],
        )

        print(f"\nSeed {seed} - group results")

        for name, value in metrics.items():
            if value is None:
                print(f"{name}: undefined")
            else:
                print(f"{name}: {value:.4f}")

        print("Counts:", counts)
        print("Checks:", checks, "\n")

        results.append({
            "seed": seed,
            "metrics": metrics,
            "counts": counts,
            "checks": checks,
            "folds": folds,
            "group_predictions": [
                {
                    "group_id": int(group),
                    "actual": str(group_y[index]),
                    "positive_probability": float(
                        probability[first[index]]
                    ),
                    "decision": str(decisions[index]),
                }
                for index, group in enumerate(unique_groups)
            ],
        })

    # 7. Önceden belirlediğimiz kuralla sonucu bildir.
    statuses = [
        status
        for result in results
        for status in result["checks"].values()
    ]

    if "FAIL" in statuses:
        outcome = "TRY_C1"
    elif "TOLERANCE" in statuses:
        outcome = "KEEP_B1_WITH_TOLERANCE"
    else:
        outcome = "KEEP_B1"

    report_path = REPORTS / "B1_threshold_stability.json"

    save_json(
        report_path,
        {
            "plan": plan,
            "results": results,
            "outcome": outcome,
        },
    )

    print("Outcome:", outcome)
    print("Saved:", report_path)
    print(
        "Final test predictions: not generated. "
        "Existing models: unchanged."
    )


if __name__ == "__main__":
    main()