from collections import Counter, defaultdict
from pathlib import Path
import sqlite3


def prepare_training_data():
    project_dir = Path(__file__).resolve().parent
    database_path = (
        project_dir / "data" / "processed" / "reviews.sqlite3"
    )

    connection = sqlite3.connect(
        database_path.as_uri() + "?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute("""
            SELECT
                r.review_id,
                r.content,
                r.score,
                lp.language_candidate,
                ld.review_label AS language_review,
                ld.reviewed_content AS language_review_content,
                sd.manual_sentiment AS short_sentiment,
                sd.review_status AS short_status,
                sd.reviewed_content AS short_review_content,
                ed.manual_sentiment AS emoji_sentiment,
                ed.review_status AS emoji_status
            FROM reviews AS r

            LEFT JOIN language_predictions AS lp
                ON r.review_id = lp.review_id
                AND lp.model_name = ?
                AND lp.preparation_version = ?

            LEFT JOIN language_review_decisions AS ld
                ON r.review_id = ld.review_id
                AND ld.model_name = lp.model_name
                AND ld.preparation_version = lp.preparation_version

            LEFT JOIN short_text_review_decisions AS sd
                ON r.review_id = sd.review_id

            LEFT JOIN letter_free_review_decisions AS ed
                ON r.review_id = ed.review_id

            WHERE r.exclusion_reason IS NULL
            ORDER BY r.review_id
        """, ("lid.176.bin", "v1")).fetchall()
    finally:
        connection.close()

    candidates = []
    source_counts = Counter()
    held_back = 0

    for row in rows:
        review_id = row["review_id"]
        content = row["content"]
        score = row["score"]

        if score not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid score: {review_id!r}")

        if row["language_candidate"] is None:
            raise ValueError(
                f"Missing language result: {review_id!r}"
            )

        # Kaydedilen incelemeden sonra metin değişmiş mi?
        for field in (
            "language_review_content",
            "short_review_content",
        ):
            reviewed_content = row[field]

            if reviewed_content is not None:
                if reviewed_content != content:
                    raise ValueError(
                        f"Content changed after review: {review_id!r}"
                    )

        if score <= 2:
            rating_sentiment = "negative"
        elif score == 3:
            rating_sentiment = "neutral"
        else:
            rating_sentiment = "positive"

        has_letters = any(
            character.isalpha() for character in content
        )

        # Önce manuel inceleme kararlarını dikkate al.
        if not has_letters:
            if row["emoji_status"] != "candidate":
                held_back += 1
                continue

            sentiment = row["emoji_sentiment"]
            source = "reviewed_letter_free"

        elif row["language_review"] is not None:
            if row["language_review"] != "english":
                held_back += 1
                continue

            sentiment = rating_sentiment
            source = "reviewed_english"

        elif row["language_candidate"] == "short_text":
            if row["short_status"] != "candidate":
                held_back += 1
                continue

            sentiment = row["short_sentiment"]
            source = "reviewed_short_text"

        elif row["language_candidate"] == "en":
            sentiment = rating_sentiment
            source = "predicted_english"

        else:
            held_back += 1
            continue

        # Manuel aday kararının yıldızla uyumunu tekrar kontrol et.
        if sentiment != rating_sentiment:
            raise ValueError(
                f"Candidate label mismatch: {review_id!r}"
            )

        # Model kopyasında yalnızca ten rengini eşitle.
        model_content = "".join(
            character
            for character in content
            if not 0x1F3FB <= ord(character) <= 0x1F3FF
        )

        # Bölmede birlikte tutulacak metinlerin anahtarı.
        group_key = " ".join(model_content.casefold().split())

        candidates.append({
            "review_id": review_id,
            "content": model_content,
            "sentiment": sentiment,
            "group_key": group_key,
            "source": source,
        })
        source_counts[source] += 1

    if len(candidates) + held_back != len(rows):
        raise ValueError("Record counts do not match.")

    candidate_ids = {
        row["review_id"] for row in candidates
    }

    if len(candidate_ids) != len(candidates):
        raise ValueError("Duplicate review IDs in candidate data.")

    label_counts = Counter(
        row["sentiment"] for row in candidates
    )

    group_labels = defaultdict(set)

    for row in candidates:
        group_labels[row["group_key"]].add(row["sentiment"])

    conflicting_groups = sum(
        len(labels) > 1
        for labels in group_labels.values()
    )

    print("Candidate data summary:")
    print("Analysis records:", len(rows))
    print("Selected candidates:", len(candidates))
    print("Held back:", held_back)

    print("\nCandidates by source:")
    for source in (
        "predicted_english",
        "reviewed_english",
        "reviewed_short_text",
        "reviewed_letter_free",
    ):
        print(f"{source}: {source_counts[source]}")

    print("\nSentiment distribution:")
    for sentiment in ("negative", "neutral", "positive"):
        print(f"{sentiment}: {label_counts[sentiment]}")

    print("\nText groups:", len(group_labels))
    print(
        "Additional records within text groups:",
        len(candidates) - len(group_labels),
    )
    print("Groups with conflicting labels:", conflicting_groups)

    return candidates


if __name__ == "__main__":
    prepare_training_data()