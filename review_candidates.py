from pathlib import Path
import sqlite3


def remove_skin_tones(text):
    return "".join(
        character
        for character in text
        if not 0x1F3FB <= ord(character) <= 0x1F3FF
    )


def reviewed_sentiment(content):
    compact = "".join(content.split())

    repeated_patterns = {
        "👍": "positive",
        "❤️": "positive",
        "😊": "positive",
        "👎": "negative",
        "😅": "unclear",
        "😍": "positive",
        "👌": "positive",
    }

    for pattern, sentiment in repeated_patterns.items():
        repeat_count = len(compact) // len(pattern)

        if repeat_count > 0 and compact == pattern * repeat_count:
            return sentiment

    exact_labels = {
        "😒🙄": "negative",
        "👀": "unclear",
        "450/500": "unclear",
        "😕": "negative",
        "🤦‍♂️🤦‍♂️": "negative",
        "🌹": "unclear",
        "🤣": "unclear",
        "🤯🤯🤯": "unclear",
        "🙏": "unclear",
        "🥴": "unclear",
        "🙂🙂": "unclear",
                "😎👍": "positive",
        "👍👍🥰": "positive",
        "👍👍👍👌👌👌👌": "positive",
        "💯🤩": "positive",
        "🤩🤩🤩👏👏👍👍": "positive",
        "💯❤️": "positive",
        "🥰😘": "positive",
        "😚😄": "positive",
        "👍👌": "positive",
        "👍🤝": "positive",
        "🌹🙏": "positive",
        "♥️💝💓🍁🍂": "positive",
                "☺🙋👍": "positive",
        "💋💋💋": "positive",
        "🥳🥳🥳🥳": "positive",
        ",🥺": "unclear",
        "👋👋👋👋": "unclear",
        "😳😳": "unclear",
        "🤫": "unclear",
        "💪💪": "unclear",
        "🌟🌟": "unclear",
        "😁😁": "unclear",
                "😊😊😊😊😊😊😊😊😊😊😊😘": "positive",
        "🙏🙏🙏🌷🙏🙏🙏": "positive",
        "☠️": "negative",
        "ং্": "unclear",
        "🐕‍🦺": "unclear",
        "6731236": "unclear",
        "55555": "unclear",
        "5⅞7": "unclear",
        "@#₹$%&-+()": "unclear",
        "✴️✴️✴️": "unclear",
        "💬": "unclear",
        "1/2": "unclear",
        "🍵🍵🍵🍵🍵🍵": "unclear",
        "24111957": "unclear",
        "5050": "unclear",
        "24": "unclear",
        "💐": "unclear",
        "🇮🇳🇮🇳🇮🇳": "unclear",
        "👆👍👍": "unclear",
        "😈": "unclear",
        "000067.": "unclear",
    }

    return exact_labels.get(compact)


# 1. Veritabanının yolunu oluştur.
project_dir = Path(__file__).resolve().parent
database_path = project_dir / "data" / "processed" / "reviews.sqlite3"

# 2. Veritabanını yalnızca okuma modunda aç.
connection = sqlite3.connect(
    database_path.as_uri() + "?mode=ro",
    uri=True,
)

try:
    rows = connection.execute("""
        SELECT content, score
        FROM reviews
        WHERE exclusion_reason IS NULL
        ORDER BY review_id
    """).fetchall()
finally:
    connection.close()


# 3. Harf içermeyen yorumları, ten rengi varyasyonlarını
# eşitleyerek özetle. Orijinal kayıtlar değiştirilmez.
summary = {}

for content, score in rows:
    if any(character.isalpha() for character in content):
        continue

    content = remove_skin_tones(content)

    if content not in summary:
        summary[content] = {
            "negative": 0,
            "neutral": 0,
            "positive": 0,
        }

    if score <= 2:
        rating_group = "negative"
    elif score == 3:
        rating_group = "neutral"
    else:
        rating_group = "positive"

    summary[content][rating_group] += 1


# 4. İçerikleri ve yıldız dağılımlarını göster.
letter_free_count = sum(
    sum(counts.values())
    for counts in summary.values()
)

print("Letter-free review count:", letter_free_count)
print("Distinct contents:", len(summary))

for content, counts in summary.items():
    print(f"\nContent: {content!r}")
    print(
        f"Negative stars: {counts['negative']} | "
        f"Neutral stars: {counts['neutral']} | "
        f"Positive stars: {counts['positive']}"
    )


# 5. Onaylanan manuel kararları uygula.
candidate_count = 0
mismatch_count = 0
unclear_count = 0
unreviewed_count = 0

for content, counts in summary.items():
    sentiment = reviewed_sentiment(content)
    total = sum(counts.values())

    if sentiment is None:
        unreviewed_count += total
    elif sentiment == "unclear":
        unclear_count += total
    else:
        candidate_count += counts[sentiment]
        mismatch_count += total - counts[sentiment]


# 6. Sonuçları göster.
print("\nManual review decisions:")
print("Training candidates:", candidate_count)
print("Rating-content mismatches:", mismatch_count)
print("Unclear content:", unclear_count)
print("Not reviewed yet:", unreviewed_count)

classified_total = (
    candidate_count
    + mismatch_count
    + unclear_count
    + unreviewed_count
)

print("Total classified records:", classified_total)

if classified_total != letter_free_count:
    raise ValueError("Review counts do not match.")


# 7. Henüz değerlendirilmemiş içerikleri göster.
print("\nContents still awaiting review:")

for content in summary:
    if reviewed_sentiment(content) is None:
        print(repr(content))