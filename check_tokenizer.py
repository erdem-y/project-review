from text_features import tokenize_review
from prepare_training_data import prepare_training_data


# 1. Sonucunu önceden bildiğimiz örnekleri kontrol et.
examples = [
    ("Not good 😡", ["not", "good", "😡"]),
    ("Good👍app", ["good", "👍", "app"]),
    ("👍🏽👍🏽", ["👍", "👍"]),
    ("❤️", ["❤️"]),
    ("☺️😃", ["☺️", "😃"]),
    ("🤦‍♂️", ["🤦‍♂️"]),
    ("I can't connect!", ["i", "can't", "connect"]),
    ("I can’t connect!", ["i", "can't", "connect"]),
    ("It's not working", ["it's", "not", "working"]),
    ("Error 403", ["error", "403"]),
]

failed = 0

print("Tokenizer checks:")

for text, expected in examples:
    actual = tokenize_review(text)
    passed = actual == expected

    print(f"\nText: {text!r}")
    print("Tokens:", actual)
    print("Result:", "PASS" if passed else "FAIL")

    if not passed:
        failed += 1
        print("Expected:", expected)

print(f"\nPassed: {len(examples) - failed}/{len(examples)}")


# 2. Aday veride 't' ve 'g' token'larının geçtiği metinleri bul.
# Bu bölüm hiçbir kayıt veya etiketi değiştirmez.
candidates = prepare_training_data()

for target in ("t", "g"):
    matching_records = 0
    distinct_texts = {}

    for row in candidates:
        text = row["content"]

        if target in tokenize_review(text):
            matching_records += 1
            distinct_texts.setdefault(text, row["review_id"])

    print(f"\nToken: {target!r}")
    print("Candidate records containing token:", matching_records)
    print("Distinct texts:", len(distinct_texts))
    print("First 10 distinct examples:")

    for text, review_id in list(distinct_texts.items())[:10]:
        print(f"\nID: {review_id}")
        print("Text:", repr(text))
        print("Tokens:", tokenize_review(text))

if failed:
    print("\nSome checks failed. Inspect them before changing the model.")
else:
    print("\nAll predefined tokenizer checks passed.")