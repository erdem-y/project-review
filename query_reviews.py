from pathlib import Path
import sqlite3


project_dir = Path(__file__).resolve().parent
database_path = project_dir / "data" / "processed" / "reviews.sqlite3"

connection = sqlite3.connect(database_path)

try:
    search_term = "bluetooth"

    results = connection.execute("""
        SELECT
            r.review_id,
            r.content,
            r.score,
            lp.language_candidate,
            lp.language_score
        FROM reviews AS r
        JOIN language_predictions AS lp
            ON r.review_id = lp.review_id
        WHERE r.exclusion_reason IS NULL
          AND lp.model_name = ?
          AND lp.preparation_version = ?
        ORDER BY r.review_id
        LIMIT 5
    """, ("lid.176.bin", "v1")).fetchall()

    for review_id, content, score, candidate, language_score in results:
        print(f"\nID: {review_id} | Stars: {score}")
        print(f"Language candidate: {candidate} | Score: {language_score}")
        print(content)
        
    missing_count = connection.execute("""
        SELECT COUNT(*)
        FROM reviews AS r
        LEFT JOIN language_predictions AS lp
            ON r.review_id = lp.review_id
           AND lp.model_name = ?
           AND lp.preparation_version = ?
        WHERE r.exclusion_reason IS NULL
          AND lp.review_id IS NULL
    """, ("lid.176.bin", "v1")).fetchone()[0]

    print("\nReviews without a stored language result:", missing_count)

    labels = connection.execute("""
        SELECT label_id, display_name
        FROM topic_labels
        ORDER BY label_id
    """).fetchall()

    print(f"\nTopic labels: {len(labels)}")

    for label_id, display_name in labels:
        print(f"{label_id}: {display_name}")

finally:
    connection.close()