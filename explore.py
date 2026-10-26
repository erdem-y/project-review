from pathlib import Path
import unicodedata

import fasttext
import pandas as pd

import sqlite3

# 1. Kaynak dosyanın yolunu oluştur.
project_dir = Path(__file__).resolve().parent
file_path = (
    project_dir
    / "data"
    / "raw"
    / "Top10_global_apps_by_downloads_playstore_reviews.csv"
)

# 2. Kaynak metinleri koruyarak CSV'yi oku.
reviews = pd.read_csv(
    file_path,
    encoding="utf-8",
    keep_default_na=False,
)

print("Columns:", reviews.columns.tolist())

print("\nReview counts by app:")
print(reviews["app"].value_counts(dropna=False))


# 3. Yalnızca Zoom yorumlarını seç.
zoom_reviews = reviews.loc[
    reviews["app"] == "Zoom Cloud Meetings"
].copy()

print("\nZoom review count before exclusion:", len(zoom_reviews))


# 4. İnceleyip çıkarmaya karar verdiğimiz tek kaydı dışla.
excluded_review_id = "031e47e1-8d42-47ac-921b-8be91a0fcd34"

zoom_analysis = zoom_reviews.loc[
    zoom_reviews["reviewId"] != excluded_review_id
].copy()

print("\nExcluded rows:", len(zoom_reviews) - len(zoom_analysis))
print("Remaining Zoom rows:", len(zoom_analysis))


# 5. Eksik veya yalnızca boşluk içeren yorumları kontrol et.
content = zoom_analysis["content"].astype("string")

empty_content = (
    content.isna()
    | content.str.strip().eq("").fillna(False)
)

print("\nMissing or whitespace-only content count:")
print(empty_content.sum())

print("\nRows with missing or whitespace-only content:")
print(
    zoom_analysis.loc[
        empty_content,
        ["reviewId", "content", "score"],
    ].to_string(index=False)
)


# 6. Yıldız dağılımını göster.
print("\nZoom rating counts after exclusion:")
print(
    zoom_analysis["score"]
    .value_counts(dropna=False)
    .sort_index()
)


# 7. Kimlik ve metin tekrarlarını incele; kayıt silme.
print("\nRepeated review IDs — additional rows:")
print(zoom_analysis["reviewId"].duplicated().sum())

print("\nRepeated exact texts — additional rows:")
print(zoom_analysis["content"].duplicated().sum())

print("\n10 most frequent exact texts:")
print(zoom_analysis["content"].value_counts().head(10))

print("\nRatings for the exact text 'Good':")
print(
    zoom_analysis.loc[
        zoom_analysis["content"] == "Good",
        "score",
    ].value_counts().sort_index()
)

rating_counts_per_text = (
    zoom_analysis.groupby("content")["score"].nunique()
)

print("\nDistinct texts associated with multiple star ratings:")
print((rating_counts_per_text > 1).sum())


# 8. Orijinal yorumların karakter uzunluklarını incele.
text_lengths = content.str.len()

print("\nReview length in characters:")
print(text_lengths.describe())

shortest_indices = text_lengths.nsmallest(10).index

print("\n10 shortest reviews:")
print(
    zoom_analysis.loc[
        shortest_indices,
        ["content", "score"],
    ].to_string(index=False)
)


# 9. Orijinal yorumlarda harf bulunup bulunmadığını kontrol et.
has_letters = zoom_analysis["content"].map(
    lambda text: any(character.isalpha() for character in text)
)

print("\nReviews containing at least one letter:")
print(has_letters.sum())

print("\nReviews containing no letters:")
print((~has_letters).sum())

print("\nFirst 20 reviews containing no letters:")
print(
    zoom_analysis.loc[
        ~has_letters,
        ["content", "score"],
    ].head(20).to_string(index=False)
)


# 10. Hazır dil modelini yükle.
language_model = fasttext.load_model(
    str(project_dir / "models" / "lid.176.bin")
)


def inspect_language(text):
    result = {
        "language_candidate": "undetermined",
        "language_score": None,
        "language_margin": None,
    }

    # Dil tespiti için ayrı bir kopya hazırla.
    normalized = unicodedata.normalize("NFKC", text)

    if not any(character.isalpha() for character in normalized):
        result["language_candidate"] = "not_applicable"
        return pd.Series(result)

    # Harf, birleşen işaret, sayı ve noktalamayı koru.
    # Diğer karakterleri dil kopyasında boşlukla değiştir.
    language_characters = []

    for character in normalized:
        if character in ("\u200d", "\ufe0e", "\ufe0f"):
            continue

        category = unicodedata.category(character)

        if character.isspace() or category[0] in "LMNP":
            language_characters.append(character)
        else:
            language_characters.append(" ")

    language_text = (
        " ".join("".join(language_characters).split()).lower()
    )

    # Yalnızca harf içeren, boşlukla ayrılmış parçaları say.
    word_count = sum(
        any(character.isalpha() for character in part)
        for part in language_text.split()
    )

    if word_count < 6:
        result["language_candidate"] = "short_text"
        return pd.Series(result)

    labels, scores = language_model.predict(language_text, k=2)

    if len(scores) == 0:
        return pd.Series(result)

    result["language_candidate"] = labels[0].removeprefix("__label__")
    result["language_score"] = float(scores[0])

    if len(scores) >= 2:
        result["language_margin"] = float(scores[0] - scores[1])

    return pd.Series(result)


# 11. Fonksiyonu çalıştır ve sonuçları tabloya ekle.
print("\nStarting language detection...")

language_results = zoom_analysis["content"].apply(inspect_language)
zoom_analysis = zoom_analysis.join(language_results)

print("\nLanguage candidates — not verified labels:")
print(
    zoom_analysis["language_candidate"]
    .value_counts()
    .to_string()
)


# 12. Hesaplanabilen en küçük skor farklarını göster.
print("\n20 reviews with the smallest available score margins:")
print(
    zoom_analysis.loc[
        zoom_analysis["language_margin"].notna(),
        ["content", "language_candidate", "language_score", "language_margin"],
    ]
    .sort_values("language_margin")
    .head(20)
    .to_string(index=False)
)


# 13. Başka dil adayı alan farklı yorumlardan örnek göster.
# Tekrarları yalnızca inceleme tablosunda azalt; asıl kayıtları koru.
long_non_english = zoom_analysis.loc[
    ~zoom_analysis["language_candidate"].isin(
        ["en", "short_text", "not_applicable", "undetermined"]
    ),
    ["content", "language_candidate", "language_score", "language_margin"],
].drop_duplicates(subset=["content"])

print("\nUp to 20 distinct long reviews predicted as non-English:")
print(
    long_non_english
    .sample(n=min(20, len(long_non_english)), random_state=42)
    .to_string(index=False)
)
# 14. Dil sonuçlarını SQLite'a kaydet.
database_path = project_dir / "data" / "processed" / "reviews.sqlite3"

model_name = "lid.176.bin"
preparation_version = "v1"

prediction_records = []

for row in zoom_analysis.itertuples(index=False):
    score = (
        None
        if pd.isna(row.language_score)
        else float(row.language_score)
    )

    margin = (
        None
        if pd.isna(row.language_margin)
        else float(row.language_margin)
    )

    prediction_records.append((
        row.reviewId,
        model_name,
        preparation_version,
        row.language_candidate,
        score,
        margin,
    ))

connection = sqlite3.connect(database_path)
connection.execute("PRAGMA foreign_keys = ON")

try:
    connection.executemany("""
        INSERT INTO language_predictions (
            review_id,
            model_name,
            preparation_version,
            language_candidate,
            language_score,
            language_margin
        )
        VALUES (?, ?, ?, ?, ?, ?)

        ON CONFLICT (
            review_id,
            model_name,
            preparation_version
        )
        DO UPDATE SET
            language_candidate = excluded.language_candidate,
            language_score = excluded.language_score,
            language_margin = excluded.language_margin
    """, prediction_records)

    connection.commit()

    saved_count = connection.execute("""
        SELECT COUNT(*)
        FROM language_predictions
        WHERE model_name = ?
          AND preparation_version = ?
    """, (model_name, preparation_version)).fetchone()[0]

    print("\nStored language results for this model/version:", saved_count)

finally:
    connection.close()
