from collections import Counter
from pathlib import Path
import sqlite3


project_dir = Path(__file__).resolve().parent
database_path = project_dir / "data" / "processed" / "reviews.sqlite3"

connection = sqlite3.connect(
    database_path.as_uri() + "?mode=ro",
    uri=True,
)

try:
    rows = connection.execute("""
        SELECT
            r.review_id,
            r.content,
            lp.language_candidate,
            d.review_status
        FROM reviews AS r
        LEFT JOIN language_predictions AS lp
            ON r.review_id = lp.review_id
            AND lp.model_name = ?
            AND lp.preparation_version = ?
        LEFT JOIN letter_free_review_decisions AS d
            ON r.review_id = d.review_id
        WHERE r.exclusion_reason IS NULL
        ORDER BY r.review_id
    """, ("lid.176.bin", "v1")).fetchall()
finally:
    connection.close()


language_counts = Counter()
manual_counts = Counter()

letter_count = 0
letter_free_count = 0
unexpected_decisions = 0

for review_id, content, language_candidate, review_status in rows:
    language_counts[
        language_candidate if language_candidate is not None else "missing"
    ] += 1

    has_letters = any(
        character.isalpha() for character in content
    )

    if has_letters:
        letter_count += 1

        if review_status is not None:
            unexpected_decisions += 1
    else:
        letter_free_count += 1
        manual_counts[
            review_status if review_status is not None else "missing"
        ] += 1


print("Analysis records:", len(rows))
print("With letters:", letter_count)
print("Without letters:", letter_free_count)

print("\nStored language candidates — not verified labels:")
for candidate, count in language_counts.most_common():
    print(f"{candidate}: {count}")

print("\nManual decisions for letter-free reviews:")
for status in ("candidate", "mismatch", "unclear", "missing"):
    print(f"{status}: {manual_counts[status]}")

print(
    "\nReviews with letters but a letter-free decision:",
    unexpected_decisions,
)

if letter_count + letter_free_count != len(rows):
    raise ValueError("Harf kontrolündeki kayıt toplamları uyuşmuyor.")

if sum(manual_counts.values()) != letter_free_count:
    raise ValueError("Manuel karar toplamları uyuşmuyor.")

    # İngilizce dışında dil adayı alan uzun yorumları incele.
connection = sqlite3.connect(
    database_path.as_uri() + "?mode=ro",
    uri=True,
)

try:
    language_review_rows = connection.execute("""
        SELECT
            r.content,
            lp.language_candidate,
            COUNT(*) AS record_count
        FROM reviews AS r
        JOIN language_predictions AS lp
            ON r.review_id = lp.review_id
        WHERE r.exclusion_reason IS NULL
            AND lp.model_name = ?
            AND lp.preparation_version = ?
            AND lp.language_candidate NOT IN (
                'en',
                'short_text',
                'not_applicable',
                'undetermined'
            )
        GROUP BY r.content, lp.language_candidate
        ORDER BY MIN(r.review_id)
    """, ("lid.176.bin", "v1")).fetchall()
finally:
    connection.close()


print("\nOther-language candidates for inspection:")
print(
    "Total records:",
    sum(count for _, _, count in language_review_rows),
)
print("Distinct text-language pairs:", len(language_review_rows))

# İlk incelemede en fazla 10 farklı içerik göster.
for number, (content, candidate, count) in enumerate(
    language_review_rows[40:],
    start=41,
):
    print(f"\nItem {number} | Candidate: {candidate} | Records: {count}")
    print(repr(content))

    # Kısa yorumları orijinal metinlerine göre grupla.
connection = sqlite3.connect(
    database_path.as_uri() + "?mode=ro",
    uri=True,
)

try:
    short_text_groups = connection.execute("""
        SELECT
            r.content,
            COUNT(*) AS record_count,
            SUM(CASE WHEN r.score <= 2 THEN 1 ELSE 0 END),
            SUM(CASE WHEN r.score = 3 THEN 1 ELSE 0 END),
            SUM(CASE WHEN r.score >= 4 THEN 1 ELSE 0 END)
        FROM reviews AS r
        JOIN language_predictions AS lp
            ON r.review_id = lp.review_id
        WHERE r.exclusion_reason IS NULL
            AND lp.model_name = ?
            AND lp.preparation_version = ?
            AND lp.language_candidate = 'short_text'
        GROUP BY r.content
        ORDER BY record_count DESC, r.content
    """, ("lid.176.bin", "v1")).fetchall()
finally:
    connection.close()


short_total = sum(
    row[1] for row in short_text_groups
)

top_groups = short_text_groups[:20]
top_total = sum(
    row[1] for row in top_groups
)

print("\nShort-text repetition summary:")
print("Total records:", short_total)
print("Distinct exact texts:", len(short_text_groups))
print("Records covered by top 20 texts:", top_total)

if short_total:
    print(f"Top 20 coverage: {top_total / short_total:.1%}")

print("\n20 most frequent short texts:")

for number, row in enumerate(top_groups, start=1):
    content, count, negative, neutral, positive = row

    print(f"\nItem {number} | Records: {count}")
    print(repr(content))
    print(
        f"Negative stars: {negative} | "
        f"Neutral stars: {neutral} | "
        f"Positive stars: {positive}"
    )