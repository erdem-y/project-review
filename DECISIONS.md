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

## 003 — Database structure and review labels

In this part, I started using SQLite in Python to store my data in a database. I chose review ID, content, score, app, and app category as the main fields.

I decided to keep information such as language predictions and user experience in separate tables and connect them with JOIN when needed. This way, I can keep model results separate from the original data. The contribution to the star average will be calculated separately when needed.

I defined topic labels to organize the review content. Each review can have between one and three different labels, selected in the order the topics first appear in the text. If more than three topics are mentioned, the system will mark that additional topics exist. This can be shown with a “...” symbol in the interface.

The same label will not be repeated within one review. Reviews can be filtered by one label or by a combination of labels.

Each understandable review will have one main sentiment based on the dominant sentiment expressed in the text. The star-based rating group will remain separate. If the content cannot be understood, it will be marked as unclear rather than neutral.

For now, I will continue with Zoom-specific labels. Adapting the system to other sectors will be a later step.