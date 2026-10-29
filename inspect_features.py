from sklearn.feature_extraction.text import TfidfVectorizer

from text_features import tokenize_review


example_texts = [
    "Good app 👍",
    "Not good 😡",
    "I can't connect",
    "Ok",
]

vectorizer = TfidfVectorizer(
    tokenizer=tokenize_review,
    token_pattern=None,
    lowercase=False,
    ngram_range=(1, 2),
)

# Yalnızca bu örneklerden özellikleri ve ağırlıkları öğren.
matrix = vectorizer.fit_transform(example_texts)
feature_names = vectorizer.get_feature_names_out()

print("Learned features:")
print(feature_names.tolist())

print("\nMatrix shape:", matrix.shape)
print("Rows = comments; columns = features")


def show_features(text, row):
    print(f"\nText: {text!r}")

    for column in row.indices:
        feature = feature_names[column]
        weight = float(row[0, column])
        print(f"{feature!r}: {weight:.3f}")


for index, text in enumerate(example_texts):
    show_features(text, matrix.getrow(index))


# Bu metinlerden yeni özellik öğrenme.
new_texts = [
    "Not good 👍",
    "Fantastic",
]

new_matrix = vectorizer.transform(new_texts)

print("\nNew comments — transform only:")

for index, text in enumerate(new_texts):
    row = new_matrix.getrow(index)
    show_features(text, row)

    if row.nnz == 0:
        print("No learned features matched.")