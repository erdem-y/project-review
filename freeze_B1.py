from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "B1_frozen.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    dev_path = ROOT / "reports" / "B1_dev.json"
    stability_path = (
        ROOT / "reports" / "B1_threshold_stability.json"
    )

    dev = read_json(dev_path)
    stability = read_json(stability_path)

    # 1. Beklediğimiz değerlendirme sonuçlarını kontrol et.
    if (
        dev["experiment"] != "B1_dev"
        or dev["final_test_scored"] is not False
    ):
        raise ValueError(
            "Expected the B1 development report "
            "without final test scoring."
        )

    if stability["outcome"] not in (
        "KEEP_B1",
        "KEEP_B1_WITH_TOLERANCE",
    ):
        raise ValueError(
            "B1 did not pass the agreed selection rule."
        )

    if stability["plan"]["final_test_scored"] is not False:
        raise ValueError(
            "Unexpected final test status in the stability report."
        )

    for report in (dev, stability["plan"]):
        if (report["lower"], report["upper"]) != (0.50, 0.65):
            raise ValueError("Unexpected decision thresholds.")

    if (
        dev["training_csv_sha256"]
        != stability["plan"]["training_csv_sha256"]
    ):
        raise ValueError(
            "Development and stability reports "
            "refer to different training data."
        )

    sklearn_version = version("scikit-learn")

    if any(
        report["sklearn_version"] != sklearn_version
        for report in (dev, stability["plan"])
    ):
        raise ValueError(
            "scikit-learn version changed after evaluation."
        )

    # 2. Dosyaları development raporundaki hash'lerle karşılaştır.
    sources = {
        "model": (
            "models/sentiment_B1.joblib",
            "model_sha256",
        ),
        "tokenizer": (
            "text_features.py",
            "tokenizer_sha256",
        ),
        "training_data": (
            "data/processed/training_B1_v3.csv",
            "training_csv_sha256",
        ),
        "evaluation_labels": (
            "data/processed/evaluation_v1/labelingyeni.csv",
            "labels_csv_sha256",
        ),
        "selection_manifest": (
            "data/processed/evaluation_v1/selection_manifest.json",
            "manifest_sha256",
        ),
    }

    files = {}

    for name, (relative_path, hash_key) in sources.items():
        actual_hash = digest(ROOT / relative_path)

        if actual_hash != dev[hash_key]:
            raise ValueError(
                "File changed since development evaluation: "
                f"{relative_path}"
            )

        files[name] = {
            "path": relative_path,
            "sha256": actual_hash,
        }

    for name, path in (
        ("development_report", dev_path),
        ("stability_report", stability_path),
    ):
        files[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path),
        }

    # 3. Seçilen ayarları ve dosya kimliklerini hazırla.
    config = {
        "schema_version": 1,
        "model_name": "B1",
        "sklearn_version": sklearn_version,
        "positive_probability_thresholds": {
            "lower": 0.50,
            "upper": 0.65,
        },
        "decision_rule": {
            "negative": "p < lower",
            "needs_review": "lower <= p < upper",
            "positive": "p >= upper",
        },
        "input_preparation": (
            "remove U+1F3FB..U+1F3FF before the saved pipeline"
        ),
        "selection_outcome": stability["outcome"],
        "final_test_scored_at_freeze": False,
        "files": files,
    }

    # Mevcut config farklıysa sessizce üzerine yazma.
    if CONFIG_PATH.exists():
        saved = read_json(CONFIG_PATH)
        saved.pop("created_at_utc", None)

        if saved != config:
            raise ValueError(
                "Existing frozen config differs. "
                "It was not overwritten."
            )

        print("Existing frozen config verified:", CONFIG_PATH)

    else:
        config["created_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CONFIG_PATH.open("x", encoding="utf-8") as file:
            json.dump(
                config,
                file,
                indent=2,
                allow_nan=False,
            )

        print("Frozen config saved:", CONFIG_PATH)

    print("Model SHA-256:", files["model"]["sha256"])
    print("Fixed thresholds: lower=0.50, upper=0.65")
    print(
        "Training: not performed. "
        "Final test predictions: not generated."
    )


if __name__ == "__main__":
    main()