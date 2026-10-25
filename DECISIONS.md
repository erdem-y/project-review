# Project Decisions

## 001 — Use a project-specific Python environment

- Decision: I decided to use a Python virtual environment for this project.
- Reason: It keeps this project's Python packages separate from those used by other projects, helping prevent version conflicts.

## 002 — Dataset selection and language detection

I chose the Zoom app reviews for this project. I excluded one review
containing only "None" from the analysis and kept the remaining 4,999
records, including emoji-only and numeric comments.

I compared Lingua, langdetect, and fastText on sample comments and chose
fastText based on the results I inspected.

For language detection, I created a separate copy of each comment,
normalized decorative characters, removed symbols, and converted the
text to lowercase. These steps improved language predictions in the
examples I checked.

I skipped language detection for comments without letters or with fewer
than six letter-containing tokens. I kept the original comments unchanged
and did not automatically remove reviews based on language predictions.