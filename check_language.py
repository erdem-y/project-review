from lingua import LanguageDetectorBuilder


detector = LanguageDetectorBuilder.from_all_languages().build()

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

    candidates = detector.compute_language_confidence_values(text)

    for candidate in candidates[:3]:
        print(f"{candidate.language.name}: {candidate.value:.3f}")