from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException


DetectorFactory.seed = 0

examples = [
    "The app crashes whenever I join a meeting.",
    "Uygulama toplantıya katılınca kapanıyor.",
    "Good",
    "Nice app",
    "👍👍",
    "12345",
]

for text in examples:
    print(f"\nText: {text!r}")

    if not any(character.isalpha() for character in text):
        print("No letters — language detection skipped.")
        continue

    try:
        candidates = detect_langs(text)

        for candidate in candidates[:3]:
            print(f"{candidate.lang}: {candidate.prob:.3f}")

    except LangDetectException as error:
        print(f"Language detection failed: {error}")