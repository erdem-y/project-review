from collections import Counter
from pathlib import Path
import sqlite3


def normalize_for_matching(content):
    # Orijinal metni değiştirme.
    return " ".join(content.casefold().split())


def main():
    labels = {
        "good": "positive",
        "nice": "positive",
        "very good": "positive",
        "excellent": "positive",
        "nice app": "positive",
        "great": "positive",
        "good app": "positive",
        "super": "positive",
        "best": "positive",
        "awesome": "positive",
        "very good app": "positive",
        "very nice": "positive",
        "best app": "positive",
        "good 👍": "positive",
        "wonderful": "positive",
        "superb": "positive",
        "bad": "negative",
        "ok": "neutral",
    }

    project_dir = Path(__file__).resolve().parent
    database_path = (
        project_dir / "data" / "processed" / "reviews.sqlite3"
    )

    connection = sqlite3.connect(
        database_path.as_uri() + "?mode=rw",
        uri=True,
    )

    counts = Counter()
    decisions = []
    unreviewed_count = 0

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")

        rows = connection.execute("""
            SELECT r.review_id, r.content, r.score
            FROM reviews AS r
            JOIN language_predictions AS lp
                ON r.review_id = lp.review_id
            WHERE r.exclusion_reason IS NULL
                AND lp.model_name = ?
                AND lp.preparation_version = ?
                AND lp.language_candidate = 'short_text'
            ORDER BY r.review_id
        """, ("lid.176.bin", "v1")).fetchall()

        for review_id, content, score in rows:
            matching_text = normalize_for_matching(content)
            sentiment = labels.get(matching_text)

            if sentiment is None:
                unreviewed_count += 1
                continue

            if score not in (1, 2, 3, 4, 5):
                raise ValueError(
                    f"Invalid score: {review_id!r}, {score!r}"
                )

            if score <= 2:
                rating_sentiment = "negative"
            elif score == 3:
                rating_sentiment = "neutral"
            else:
                rating_sentiment = "positive"

            if sentiment == rating_sentiment:
                status = "candidate"
            else:
                status = "mismatch"

            decisions.append((
                review_id,
                content,
                sentiment,
                rating_sentiment,
                status,
            ))
            counts[status] += 1

        if len(decisions) + unreviewed_count != len(rows):
            raise ValueError("Record counts do not match.")

        connection.execute("""
            CREATE TABLE IF NOT EXISTS short_text_review_decisions (
                review_id TEXT PRIMARY KEY NOT NULL,
                reviewed_content TEXT NOT NULL,
                manual_sentiment TEXT NOT NULL CHECK (
                    manual_sentiment IN (
                        'negative', 'neutral', 'positive'
                    )
                ),
                rating_sentiment TEXT NOT NULL CHECK (
                    rating_sentiment IN (
                        'negative', 'neutral', 'positive'
                    )
                ),
                review_status TEXT NOT NULL CHECK (
                    review_status IN ('candidate', 'mismatch')
                ),
                FOREIGN KEY (review_id)
                    REFERENCES reviews(review_id)
            )
        """)

        connection.executemany("""
            INSERT INTO short_text_review_decisions (
                review_id,
                reviewed_content,
                manual_sentiment,
                rating_sentiment,
                review_status
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(review_id) DO UPDATE SET
                reviewed_content = excluded.reviewed_content,
                manual_sentiment = excluded.manual_sentiment,
                rating_sentiment = excluded.rating_sentiment,
                review_status = excluded.review_status
        """, decisions)

        connection.commit()

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print("Short-text review decisions:")
    print("Source records:", len(rows))
    print("Training candidates:", counts["candidate"])
    print("Rating-content mismatches:", counts["mismatch"])
    print("Total saved records:", len(decisions))
    print("Not reviewed yet:", unreviewed_count)


if __name__ == "__main__":
    main()