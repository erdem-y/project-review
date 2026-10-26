from pathlib import Path
import sqlite3


project_dir = Path(__file__).resolve().parent
database_path = project_dir / "data" / "processed" / "reviews.sqlite3"

database_path.parent.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(database_path)
connection.execute("PRAGMA foreign_keys = ON")


try:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY NOT NULL,
            content TEXT NOT NULL,
            score INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
            app TEXT NOT NULL,
            app_category TEXT NOT NULL,
            exclusion_reason TEXT
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS language_predictions (
            review_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            preparation_version TEXT NOT NULL,
            language_candidate TEXT NOT NULL,
            language_score REAL,
            language_margin REAL,

            PRIMARY KEY (
                review_id,
                model_name,
                preparation_version
            ),

            FOREIGN KEY (review_id)
                REFERENCES reviews(review_id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS topic_labels (
            label_id TEXT PRIMARY KEY NOT NULL,
            display_name TEXT NOT NULL UNIQUE
        )
    """)

    labels = [
        ("bluetooth", "Bluetooth"),
        ("audio", "Ses"),
        ("camera", "Kamera"),
        ("screen", "Ekran"),
        ("usage_issues", "Kullanım sorunları"),
        ("pricing", "Ücretlendirme"),
        ("general_feedback", "Genel değerlendirme"),
        ("other", "Diğer"),
        ("unclear_content", "Belirsiz içerik"),
    ]

    connection.executemany("""
        INSERT INTO topic_labels (label_id, display_name)
        VALUES (?, ?)
        ON CONFLICT(label_id) DO NOTHING
    """, labels)

    connection.commit()
finally:
    connection.close()

print("Database tables are ready.")
print("Database path:", database_path)