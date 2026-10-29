from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import random
import sqlite3

from prepare_training_data import prepare_training_data


SEED = 42

PLAN = {
    "negative": {"dev": 10, "test": 40},
    "neutral": {"dev": 10, "test": 20},
    "positive": {"dev": 10, "test": 40},
}


def select_reviews(candidates):
    groups = defaultdict(list)

    for row in candidates:
        groups[row["group_key"]].append(row)

    for key, members in groups.items():
        labels = {row["sentiment"] for row in members}

        if len(labels) != 1:
            raise ValueError(
                f"Conflicting labels in group: {key!r}"
            )

    rng = random.Random(SEED)
    selected = []

    for label, parts in PLAN.items():
        available_keys = sorted(
            key
            for key, members in groups.items()
            if members[0]["sentiment"] == label
        )

        needed = sum(parts.values())

        if len(available_keys) < needed:
            raise ValueError(
                f"{label}: need {needed} distinct groups, "
                f"but found {len(available_keys)}."
            )

        chosen_keys = rng.sample(available_keys, needed)
        offset = 0

        for split, count in parts.items():
            for key in chosen_keys[offset:offset + count]:
                members = groups[key]
                representative = rng.choice(members)

                selected.append({
                    "row": representative,
                    "split": split,
                    "group_tokens": list(key),
                    "group_review_ids": [
                        member["review_id"]
                        for member in members
                    ],
                })

            offset += count

    # Etiketleme dosyasında sınıflar bloklar hâlinde görünmesin.
    rng.shuffle(selected)

    return selected


def main():
    candidates = prepare_training_data()
    selected = select_reviews(candidates)

    project_dir = Path(__file__).resolve().parent
    database_path = (
        project_dir / "data" / "processed" / "reviews.sqlite3"
    )

    connection = sqlite3.connect(
        database_path.as_uri() + "?mode=ro",
        uri=True,
    )

    try:
        original_contents = dict(connection.execute("""
            SELECT review_id, content
            FROM reviews
            WHERE exclusion_reason IS NULL
        """).fetchall())
    finally:
        connection.close()

    # Eğitimden ayrılacak bütün kardeş kayıtları belirle.
    held_ids = {
        review_id
        for item in selected
        for review_id in item["group_review_ids"]
    }

    remaining = [
        row for row in candidates
        if row["review_id"] not in held_ids
    ]

    selected_keys = {
        tuple(item["group_tokens"])
        for item in selected
    }
    remaining_keys = {
        row["group_key"] for row in remaining
    }

    if selected_keys & remaining_keys:
        raise ValueError("Evaluation/training group overlap.")

    if len(selected) != 130 or len(selected_keys) != 130:
        raise ValueError("Expected 130 distinct selected groups.")

    manifest = {
        "seed": SEED,
        "plan": PLAN,
        "candidate_count": len(candidates),
        "selections": [
            {
                "review_id": item["row"]["review_id"],
                "split": item["split"],
                "sampling_label": item["row"]["sentiment"],
                "group_tokens": item["group_tokens"],
                "group_review_ids": item["group_review_ids"],
            }
            for item in selected
        ],
    }

    output_dir = (
        project_dir / "data" / "processed" / "evaluation_v1"
    )

    # Mevcut bir seçim veya etiketleme dosyasının üzerine yazma.
    output_dir.mkdir(exist_ok=False)

    csv_path = output_dir / "labeling.csv"
    manifest_path = output_dir / "selection_manifest.json"

    try:
        with csv_path.open(
            "x",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([
                "review_id",
                "content",
                "sentiment",
                "notes",
            ])

            for item in selected:
                review_id = item["row"]["review_id"]

                writer.writerow([
                    review_id,
                    original_contents[review_id],
                    "",
                    "",
                ])

        with manifest_path.open(
            "x",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except Exception:
        csv_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise

    print("\nEvaluation selection:")
    print("Comments to label:", len(selected))
    print(
        "Dev comments:",
        sum(item["split"] == "dev" for item in selected),
    )
    print(
        "Test comments:",
        sum(item["split"] == "test" for item in selected),
    )
    print("Records held out, including group copies:", len(held_ids))
    print("Remaining training candidates:", len(remaining))
    print("Evaluation/training group overlap: 0")

    counts = Counter(row["sentiment"] for row in remaining)

    print("\nRemaining candidate labels:")
    for label in ("negative", "neutral", "positive"):
        print(f"{label}: {counts[label]}")

    print("\nLabeling file:", csv_path)
    print("Selection manifest:", manifest_path)


if __name__ == "__main__":
    main()