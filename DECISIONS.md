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

## 004 — Manual review and initial model candidate selection

Before starting model training, I prepared the candidate data.

First, I reviewed 103 comments containing no letters, including emoji-only and numeric comments. I selected 58 records where the sentiment was clear and matched the star-based rating group. I kept the remaining records separate because their content was unclear or their sentiment did not match their rating.

For longer comments, fastText predicted 1,412 records as English. I decided to use these as candidates without manually checking every prediction, while accepting that some language predictions may be incorrect. I also reviewed all 57 records predicted as other languages and identified two English comments to include.

Another 3,427 comments were marked as short text because they contained fewer than six whitespace-separated tokens containing letters. Language detection was skipped for these comments.

To avoid reviewing every short comment individually, I examined the 20 most frequent exact texts. These covered 1,285 records, or 37.5% of the short-text group. I assigned sentiment labels to these expressions, including neutral for “Ok”.

After ignoring case differences and standardizing whitespace in a separate matching copy, the approved expressions matched 1,353 records. Of these, 1,178 matched their star-based rating group and became candidates. The remaining 175 were kept separate as mismatches.

This produced 2,650 candidate records. I preserved the original comments and ratings and stored review decisions in separate tables.

I also created 1,465 text groups so that equivalent comments can stay together during the training/test split. The split has not been performed yet. These candidates are not a fully verified gold dataset; many sentiment labels still come from star ratings.