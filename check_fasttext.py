from pathlib import Path

import fasttext


project_dir = Path(__file__).resolve().parent
model_path = project_dir / "models" / "lid.176.bin"

model = fasttext.load_model(str(model_path))

examples = [
    "The app crashes whenever I join a meeting.",
    "Uygulama toplantıya katılınca kapanıyor.",
    "Good",
    "Nice app",
    "👍👍",
    "12345",
    "Super",
    "Good app",
    "Worst app teams is much better everytime there is some problem or the other worst app never download it",
    "I can't download it... It always says pending and even if it'll take an hour it won't download... Why is it?",
    "Bad performance ever Should be fixed your glitch problem Every time zoom meeting end automatically",
    "Excellent",
    "ok",
    "It's great",
    "গুড",
    "kamu buat cewe aku ngambek! bintang 1 😠.",
    "Kk and 86j",
    "Bad",
    "👉 बीएड की इंटर्नशिप रिकॉर्ड की सभी डायरियो पर साइंस स्टूडेंट लाइट ग्रीन, आर्ट्स स्टूडेंट पिंक कलर और कॉमर्स स्टूडेंट लाइट ब्लू कलर का कवर चढ़ावे और सभी डायरियो पर इंटर्नशिप स्कूल के प्रिंसिपल के सिग्नेचर होने चाहिए और मूल्याकन प्रपत्र और इंटर्नशिप पूर्णता प्रपत्र साथ लेकर आये इसमे किसी प्रकार की कमी होने पर रिकॉर्ड जमा नही होगा -- प्रिंसिपल",
    "oo",
]

for text in examples:
    print(f"\nText: {text!r}")

    if not any(character.isalpha() for character in text):
        print("No letters — language detection skipped.")
        continue

    labels, scores = model.predict(text, k=3)

    for label, score in zip(labels, scores):
        language = label.removeprefix("__label__")
        print(f"{language}: {score:.3f}")