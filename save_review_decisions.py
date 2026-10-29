from collections import Counter
from pathlib import Path
import sqlite3

from review_candidates import remove_skin_tones, reviewed_sentiment


def main():
    project_dir = Path(__file__).resolve().parent
    database_path = (
        project_dir / "data" / "processed" / "reviews.sqlite3"
    )

    # Yalnızca mevcut veritabanını aç; yanlış yolda yenisini oluşturma.
    connection = sqlite3.connect(
        database_path.as_uri() + "?mode=rw",
        uri=True,
    )

    try:
        # Okuma ve yazmayı aynı işlem içinde yap.
        connection.execute("BEGIN IMMEDIATE")

        rows = connection.execute("""
            SELECT review_id, content, score
            FROM reviews
            WHERE exclusion_reason IS NULL
            ORDER BY review_id
        """).fetchall()

        decisions = []
        counts = Counter()

        for review_id, content, score in rows:
            if any(character.isalpha() for character in content):
                continue

            model_content = remove_skin_tones(content)
            sentiment = reviewed_sentiment(model_content)

            if sentiment is None:
                raise ValueError(
                    f"İncelenmemiş içerik: {review_id!r}, {content!r}"
                )

            if score <= 2:
                rating_sentiment = "negative"
            elif score == 3:
                rating_sentiment = "neutral"
            else:
                rating_sentiment = "positive"

            if sentiment == "unclear":
                status = "unclear"
            elif sentiment == rating_sentiment:
                status = "candidate"
            else:
                status = "mismatch"

            decisions.append(
                (review_id, sentiment, rating_sentiment, status)
            )
            counts[status] += 1

        # Son doğruladığımız dağılım değiştiyse kayıt yapma.
        expected_counts = Counter({
            "candidate": 58,
            "mismatch": 10,
            "unclear": 35,
        })

        if counts != expected_counts:
            raise ValueError(
                "Dağılım beklenen sonuçla uyuşmuyor. "
                f"Bulunan: {dict(counts)}. Hiçbir karar kaydedilmedi."
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS letter_free_review_decisions (
                review_id TEXT PRIMARY KEY NOT NULL,
                manual_sentiment TEXT NOT NULL
                    CHECK (
                        manual_sentiment IN (
                            'positive', 'negative', 'neutral', 'unclear'
                        )
                    ),
                rating_sentiment TEXT NOT NULL
                    CHECK (
                        rating_sentiment IN (
                            'positive', 'negative', 'neutral'
                        )
                    ),
                review_status TEXT NOT NULL
                    CHECK (
                        review_status IN (
                            'candidate', 'mismatch', 'unclear'
                        )
                    )
            )
        """)

        connection.executemany("""
            INSERT INTO letter_free_review_decisions (
                review_id,
                manual_sentiment,
                rating_sentiment,
                review_status
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(review_id) DO UPDATE SET
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

    print("\nSaved review decisions:")
    print("Training candidates:", counts["candidate"])
    print("Rating-content mismatches:", counts["mismatch"])
    print("Unclear content:", counts["unclear"])
    print("Total saved records:", len(decisions))


if __name__ == "__main__":
    main()