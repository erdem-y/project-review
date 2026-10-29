import re

import emoji


# Kelimeleri ve sayıları al.
# can't gibi kelimelerin içindeki kesme işaretini koru.
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def tokenize_review(text):
    # Yalnızca işlem kopyasını değiştir.
    text = text.casefold().replace("’", "'")

    # Önceki kararımız: ten rengini eşitle, tekrarları koru.
    text = "".join(
        character
        for character in text
        if not 0x1F3FB <= ord(character) <= 0x1F3FF
    )

    tokens = []
    position = 0

    for match in emoji.emoji_list(text):
        # Emojiden önceki bölümün kelimelerini al.
        before_emoji = text[position:match["match_start"]]
        tokens.extend(WORD_PATTERN.findall(before_emoji))

        # Emojiyi ayrı bir token olarak ekle.
        tokens.append(match["emoji"])

        position = match["match_end"]

    # Son emojiden sonra kalan kelimeleri al.
    remaining_text = text[position:]
    tokens.extend(WORD_PATTERN.findall(remaining_text))

    return tokens


if __name__ == "__main__":
    examples = [
        "Not good 😡",
        "Good👍app",
        "👍🏽👍🏽",
        "❤️",
        "🤦‍♂️",
        "I can’t connect!",
        "Error 403",
    ]

    for text in examples:
        print(f"\nOriginal: {text}")
        print("Tokens:", tokenize_review(text))