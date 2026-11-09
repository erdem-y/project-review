from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import argparse
import csv
import json

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "B1_frozen.json"

GENERATED = [
    "decision",
    "p_positive",
    "input_status",
    "star_text_disagreement",
]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def parse_score(value, line):
    value = value.strip()

    if not value:
        return None

    if value not in {"1", "2", "3", "4", "5"}:
        raise ValueError(
            f"CSV line {line}: score must be 1-5 or empty; "
            f"got {value!r}."
        )

    return int(value)


def disagreement(score, decision):
    if (
        score is None
        or score == 3
        or decision == "needs_review"
    ):
        return ""

    rating = "negative" if score <= 2 else "positive"

    return "true" if rating != decision else "false"


def load_input(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = file.readline()

        if not header.strip():
            raise ValueError("CSV header is missing.")

        # Yalnızca content sütunu varsa ayırıcı aramaya gerek yok.
        if any(character in header for character in ";,\t"):
            delimiter = csv.Sniffer().sniff(
                header,
                delimiters=";,\t",
            ).delimiter
        else:
            delimiter = ","

        file.seek(0)
        reader = csv.DictReader(file, delimiter=delimiter)
        fields = reader.fieldnames or []

        if (
            len(fields) != len(set(fields))
            or "content" not in fields
        ):
            raise ValueError(
                "Use unique column names and "
                "a required 'content' column."
            )

        if set(fields) & set(GENERATED):
            raise ValueError(
                "Input already contains prediction columns. "
                "Use the original input CSV."
            )

        rows = []
        scores = []

        for row in reader:
            if (
                None in row
                or any(value is None for value in row.values())
            ):
                raise ValueError(
                    "Malformed CSV record ending on "
                    f"line {reader.line_num}."
                )

            scores.append(
                parse_score(
                    row.get("score", ""),
                    reader.line_num,
                )
            )

            row.setdefault("review_id", "")
            row.setdefault("score", "")
            rows.append(row)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    output_fields = (
        fields
        + [
            name
            for name in ("review_id", "score")
            if name not in fields
        ]
        + GENERATED
    )

    return rows, scores, output_fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    input_path = args.input_csv.resolve()
    output_path = args.output.resolve()

    if input_path == output_path or output_path.exists():
        raise ValueError(
            "Choose a new output filename; "
            "existing files are not overwritten."
        )

    # 1. Sabitlenmiş model, tokenizer ve eşikleri doğrula.
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8-sig")
    )

    if (
        config["model_name"] != "B1"
        or config["schema_version"] != 1
    ):
        raise ValueError(
            "Unexpected frozen model configuration."
        )

    if version("scikit-learn") != config["sklearn_version"]:
        raise ValueError(
            "scikit-learn version does not match "
            "the frozen configuration."
        )

    for name in ("model", "tokenizer"):
        info = config["files"][name]

        if digest(ROOT / info["path"]) != info["sha256"]:
            raise ValueError(f"Frozen {name} file changed.")

    thresholds = config["positive_probability_thresholds"]
    lower = thresholds["lower"]
    upper = thresholds["upper"]

    if (lower, upper) != (0.50, 0.65):
        raise ValueError("Unexpected B1 thresholds.")

    # 2. Girdileri doğrula ve modeli yükle.
    rows, scores, fields = load_input(input_path)

    model = joblib.load(
        ROOT / config["files"]["model"]["path"]
    )

    if set(model.classes_) != {"negative", "positive"}:
        raise ValueError("Expected a binary B1 model.")

    positive_column = list(model.classes_).index("positive")

    # 3. Boş metinleri modele göndermeden işaretle.
    nonempty = []

    for index, row in enumerate(rows):
        row.update(
            decision="needs_review",
            p_positive="",
            input_status="empty",
            star_text_disagreement="",
        )

        if row["content"].strip():
            nonempty.append(index)

    # 4. Dolu metinleri küçük gruplar hâlinde işle.
    for start in range(0, len(nonempty), 512):
        indices = nonempty[start:start + 512]

        # Orijinal content korunur; model ayrı kopyayı kullanır.
        texts = [
            "".join(
                character
                for character in rows[index]["content"]
                if not 0x1F3FB <= ord(character) <= 0x1F3FF
            )
            for index in indices
        ]

        features = model.named_steps["tfidf"].transform(texts)
        has_features = (
            np.asarray(features.getnnz(axis=1)).ravel() > 0
        )

        probabilities = model.predict_proba(
            texts
        )[:, positive_column]

        if (
            not np.all(np.isfinite(probabilities))
            or np.any(
                (probabilities < 0) | (probabilities > 1)
            )
        ):
            raise ValueError("Invalid model probabilities.")

        for index, probability, known in zip(
            indices,
            probabilities,
            has_features,
        ):
            if probability < lower:
                decision = "negative"
            elif probability >= upper:
                decision = "positive"
            else:
                decision = "needs_review"

            rows[index].update(
                decision=decision,
                p_positive=format(float(probability), ".17g"),
                input_status=(
                    "not_flagged"
                    if known
                    else "no_known_features"
                ),
                star_text_disagreement=disagreement(
                    scores[index],
                    decision,
                ),
            )

    # 5. Girdi sütunlarını koruyarak sonuçları yaz.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Processed records:", len(rows))
    print(
        "Decisions:",
        dict(Counter(row["decision"] for row in rows)),
    )
    print(
        "Input status:",
        dict(Counter(row["input_status"] for row in rows)),
    )
    print("Saved:", output_path)
    print("Model and thresholds: unchanged.")


if __name__ == "__main__":
    main()