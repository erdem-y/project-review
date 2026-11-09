from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import random

from prepare_training_data import prepare_training_data


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

EVALUATION_PATH = (
    PROCESSED_DIR / "evaluation_v1" / "selection_manifest.json"
)
FOLDS_PATH = PROCESSED_DIR / "training_cv_grouped_v2.json"

OUTPUT_DIR = PROCESSED_DIR / "quality_review_v1"
LABELING_PATH = OUTPUT_DIR / "labeling.csv"
SAMPLE_PATH = OUTPUT_DIR / "sample_manifest.json"

SEED = 42
PLAN = {
    "negative": 34,
    "neutral": 33,
    "positive": 33,
}


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    # Tekrar çalıştırıldığında doldurulmuş etiketleri koru.
    if LABELING_PATH.exists() or SAMPLE_PATH.exists():
        raise FileExistsError(
            "A quality-review file already exists. "
            "Existing files were not changed."
        )

    candidates = prepare_training_data()
    evaluation = read_json(EVALUATION_PATH)
    cv_manifest = read_json(FOLDS_PATH)

    held_keys = {
        tuple(row["group_tokens"])
        for row in evaluation["selections"]
    }

    train = [
        row
        for row in candidates
        if tuple(row["group_key"]) not in held_keys
    ]

    # A1/A2 ile aynı eğitim havuzunu kullandığımızı doğrula.
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

    if fingerprint != cv_manifest["training_fingerprint"]:
        raise ValueError("Training data differs from A1/A2.")

    groups = defaultdict(list)

    for row in train:
        groups[tuple(row["group_key"])].append(row)

    pools = {label: [] for label in PLAN}

    for key, rows in groups.items():
        labels = {row["sentiment"] for row in rows}

        if len(labels) != 1:
            raise ValueError("Conflicting labels within a group.")

        label = next(iter(labels))
        pools[label].append(key)

    rng = random.Random(SEED)
    selected = []

    for label, count in PLAN.items():
        pool = sorted(pools[label])

        if len(pool) < count:
            raise ValueError(f"Not enough groups for {label}.")

        selected_keys = rng.sample(pool, count)

        for key in selected_keys:
            members = sorted(
                groups[key],
                key=lambda row: row["review_id"],
            )
            representative = rng.choice(members)

            selected.append({
                "review_id": representative["review_id"],
                "content": representative["content"],
                "sampling_label": label,
                "source": representative["source"],
                "group_tokens": list(key),
                "group_review_ids": [
                    row["review_id"] for row in members
                ],
            })

    # Etiket sınıfları dosyada bloklar halinde görünmesin.
    rng.shuffle(selected)

    selected_keys = {
        tuple(row["group_tokens"])
        for row in selected
    }

    if len(selected) != 100 or len(selected_keys) != 100:
        raise ValueError("Expected 100 records from 100 groups.")

    if selected_keys & held_keys:
        raise ValueError("Sample overlaps with dev/test groups.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # UTF-8 BOM ve noktalı virgül, Excel'de açmayı kolaylaştırır.
    with LABELING_PATH.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "review_id",
                "content",
                "manual_sentiment",
                "language_status",
                "notes",
            ],
            delimiter=";",
        )
        writer.writeheader()

        for row in selected:
            writer.writerow({
                "review_id": row["review_id"],
                "content": row["content"],
                "manual_sentiment": "",
                "language_status": "",
                "notes": "",
            })

    # Eski etiketleri karşılaştırma için ayrı sakla.
    with SAMPLE_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "version": "quality_review_v1",
                "seed": SEED,
                "plan": PLAN,
                "training_fingerprint": fingerprint,
                "sampling_unit": "text_group",
                "selections": selected,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nQuality review sample:")
    print("Records to review:", len(selected))
    print("Distinct text groups:", len(selected_keys))
    print(
        "Sampling labels:",
        dict(Counter(row["sampling_label"] for row in selected)),
    )
    print("Dev/test group overlap: 0")
    print("\nLabeling file:", LABELING_PATH)
    print("Sample manifest:", SAMPLE_PATH)


if __name__ == "__main__":
    main()