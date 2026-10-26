from pathlib import Path
import sqlite3

import pandas as pd


project_dir = Path(__file__).resolve().parent

source_path = (
    project_dir
    / "data"
    / "raw"
    / "Top10_global_apps_by_downloads_playstore_reviews.csv"
)
database_path = project_dir / "data" / "processed" / "reviews.sqlite3"

reviews = pd.read_csv(
    source_path,
    encoding="utf-8",
    keep_default_na=False,
)

zoom_reviews = reviews.loc[
    reviews["app"] == "Zoom Cloud Meetings"
].copy()

excluded_review_id = "031e47e1-8d42-47ac-921b-8be91a0fcd34"

records = []

for row in zoom_reviews.itertuples(index=False):
    exclusion_reason = None

    if row.reviewId == excluded_review_id:
        exclusion_reason = "Manually excluded: content is only 'None'."

    records.append((
        row.reviewId,
        row.content,
        int(row.score),
        row.app,
        row.category,
        exclusion_reason,
    ))

connection = sqlite3.connect(database_path)

try:
    connection.executemany("""
        INSERT INTO reviews (
            review_id,
            content,
            score,
            app,
            app_category,
            exclusion_reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(review_id) DO NOTHING
    """, records)

    connection.commit()

    total = connection.execute(
        "SELECT COUNT(*) FROM reviews"
    ).fetchone()[0]

    included = connection.execute("""
        SELECT COUNT(*)
        FROM reviews
        WHERE exclusion_reason IS NULL
    """).fetchone()[0]

    excluded = connection.execute("""
        SELECT COUNT(*)
        FROM reviews
        WHERE exclusion_reason IS NOT NULL
    """).fetchone()[0]

    print("Stored reviews:", total)
    print("Included in analysis:", included)
    print("Excluded from analysis:", excluded)

finally:
    connection.close()