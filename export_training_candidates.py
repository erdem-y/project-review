import csv
from pathlib import Path
import sqlite3

from prepare_training_data import prepare_training_data


def main():
    candidates = prepare_training_data()

    project_dir = Path(__file__).resolve().parent
    database_path = (
        project_dir / "data" / "processed" / "reviews.sqlite3"
    )
    output_path = (
        project_dir / "data" / "processed" / "training_candidates.csv"
    )

    # Görüntülemek için orijinal yorumları da al.
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

    # Aynı token grubuna aynı numarayı ver.
    group_numbers = {}

    for row in candidates:
        key = row["group_key"]

        if key not in group_numbers:
            group_numbers[key] = len(group_numbers) + 1

    # "x": dosya zaten varsa üzerine yazmadan dur.
    with output_path.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "review_id",
            "original_content",
            "model_content",
            "candidate_sentiment",
            "source",
            "text_group",
        ])

        for row in candidates:
            writer.writerow([
                row["review_id"],
                original_contents[row["review_id"]],
                row["content"],
                row["sentiment"],
                row["source"],
                group_numbers[row["group_key"]],
            ])

    print("\nExported candidates:", len(candidates))
    print("Text groups:", len(group_numbers))
    print("File:", output_path)


if __name__ == "__main__":
    main()