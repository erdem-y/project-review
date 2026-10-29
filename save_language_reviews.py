from collections import Counter
from pathlib import Path
import sqlite3


MODEL_NAME = "lid.176.bin"
PREPARATION_VERSION = "v1"

# Bu başlangıçlar yalnızca incelediğimiz yorumları bulmak içindir.
# Yeni yorumlara otomatik etiket veren kurallar değildir.
# Her başlangıç tam olarak bir kayıtla eşleşmelidir.
REVIEWED_PREFIXES = {
    "english": (
        "No sound without wifi?",
        "Good no disturbance",
    ),
    "non_english": (
        "kamu buat cewe aku",
        "برنامج احسن",
        "Gamit ko.dati",
        "Aplikasi bagus",
        "बहुत सफल",
        "Robo de information",
        "Bahut bura app",
        "Ji bolo ji",
        "Kasih vbg",
        "इस पर विशेष",
        "SUSAH BGT",
        "Wan star Kase",
        "Bulok di makapasok",
        "जय हो विवेक",
        "ПОЖАЛУЙСТА",
        "جميل بس",
        "في مشكلة",
        "बहुत बढ़िया",
        "Wow ganda",
        "Ya app sa sauch",
        "এই সফটওয়্যার",
        "बहुत ही सुंदर",
        "👉 बीएड",
        "Download Nehi",
        "वीडियो में",
        "बहुत बहुत सुंदर",
        "Bahut bekar",
        "Download korte",
        "Wow supar",
        "King ina",
        "Gago ni ng zoom",
        "Kenapa saya",
    ),
    "mixed": (
        "Worst app....",
        "Bar bar video",
        "is very Legit",
    ),
    "unclear": (
        "I'd? ??",
        "Vfggghhh",
        "En ikn know",
        "Ok 👍p",
        "Lo lpllow",
        "Pathan Imran",
        "9 है",
        "Eeee4ee",
        "絲腌腌",
        "Namun ni",
        "Ee ee e",
        "Bla bla",
        ". . cc.",
        "5.. .m4",
        "I'm mom.",
        "गं,ळळष.",
        "YASSYWS",
        "ccrc😄",
        "ing to abhi",
        "E3👨",
    ),
}

REVIEW_NOTES = {
    "Ya app sa sauch": (
        "Low confidence: likely romanized Hindi/Urdu; "
        "exact language not verified."
    ),
    "इस पर विशेष": (
        "Possibly off-topic: vehicle service and commissions."
    ),
    "जय हो विवेक": (
        "Praise for a person; relationship to Zoom is unclear."
    ),
    "👉 बीएड": (
        "Off-topic: teacher-internship paperwork announcement."
    ),
}

EXPECTED_COUNTS = Counter({
    "english": 2,
    "non_english": 32,
    "mixed": 3,
    "unclear": 20,
})


def match_decisions(rows):
    # Kayıt sayısı değiştiyse eski kararları otomatik uygulama.
    if len(rows) != 57:
        raise ValueError(
            f"Expected 57 source records; found {len(rows)}. "
            "Nothing saved."
        )

    decisions = []
    matched_ids = set()

    for review_label, prefixes in REVIEWED_PREFIXES.items():
        for prefix in prefixes:
            matches = [
                (review_id, content)
                for review_id, content in rows
                if content.startswith(prefix)
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one match for {prefix!r}; "
                    f"found {len(matches)}. Nothing saved."
                )

            review_id, content = matches[0]

            if review_id in matched_ids:
                raise ValueError(
                    f"Duplicate decision for {review_id!r}. "
                    "Nothing saved."
                )

            matched_ids.add(review_id)

            decisions.append((
                review_id,
                MODEL_NAME,
                PREPARATION_VERSION,
                content,
                review_label,
                REVIEW_NOTES.get(prefix),
            ))

    counts = Counter(
        record[4] for record in decisions
    )

    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Unexpected decision counts: {dict(counts)}"
        )

    source_ids = {
        review_id for review_id, _ in rows
    }

    if matched_ids != source_ids:
        raise ValueError(
            "Some source records have no decision. Nothing saved."
        )

    return decisions, counts


def save_decisions(database_path):
    # Yalnızca mevcut veritabanını aç.
    connection = sqlite3.connect(
        database_path.resolve().as_uri() + "?mode=rw",
        uri=True,
    )

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")

        rows = connection.execute("""
            SELECT
                r.review_id,
                r.content
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
            ORDER BY r.review_id
        """, (MODEL_NAME, PREPARATION_VERSION)).fetchall()

        # Bütün eşleştirmeleri yazmadan önce doğrula.
        decisions, counts = match_decisions(rows)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS language_review_decisions (
                review_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                preparation_version TEXT NOT NULL,
                reviewed_content TEXT NOT NULL,
                review_label TEXT NOT NULL CHECK (
                    review_label IN (
                        'english',
                        'non_english',
                        'mixed',
                        'unclear'
                    )
                ),
                review_note TEXT,
                PRIMARY KEY (
                    review_id,
                    model_name,
                    preparation_version
                ),
                FOREIGN KEY (review_id)
                    REFERENCES reviews(review_id)
            )
        """)

        connection.executemany("""
            INSERT INTO language_review_decisions (
                review_id,
                model_name,
                preparation_version,
                reviewed_content,
                review_label,
                review_note
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                review_id,
                model_name,
                preparation_version
            )
            DO UPDATE SET
                reviewed_content = excluded.reviewed_content,
                review_label = excluded.review_label,
                review_note = excluded.review_note
        """, decisions)

        connection.commit()
        return counts

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main():
    project_dir = Path(__file__).resolve().parent
    database_path = (
        project_dir
        / "data"
        / "processed"
        / "reviews.sqlite3"
    )

    counts = save_decisions(database_path)

    print("Saved language review decisions:")

    for label in ("english", "non_english", "mixed", "unclear"):
        print(f"{label}: {counts[label]}")

    print("Total saved records:", sum(counts.values()))


if __name__ == "__main__":
    main()