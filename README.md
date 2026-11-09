# Feedback Analysis

A Python application that classifies review sentiment and flags disagreements with star ratings.

**Frozen B1 test:** 69 of 72 automatic decisions were correct; 14 of 86 positive/negative comments were sent for review. Neutral and unclear comments were evaluated separately.

**Main limitation:** neutral, unclear and mixed comments can still receive incorrect sentiment predictions.

## Approach

I trained B1 using word-based TF-IDF with single words and two-word combinations, followed by logistic regression without class weighting. A custom tokenizer preserves emojis and expressions such as `can't`.

No pretrained sentiment model was used. A pretrained fastText model was used separately during language screening.

The application returns `positive`, `negative` or `needs_review`. Star ratings are used only for comparison, never as model input.

## Data and label review

I selected 5,000 Zoom Cloud Meetings reviews from [Top 10 Global Apps - Play Store Reviews](https://www.kaggle.com/datasets/nitinchoudhary012/top-10-global-apps-play-store-reviews) on Kaggle. After excluding one comment containing only `"None"`, 4,999 records remained for analysis.

Kaggle lists the dataset license as **Unknown**. The raw CSV, local SQLite database and manual label files stay outside the repository. See [License](#license) for the scope of the project license.

Early models treated three-star training reviews as neutral. Manual review showed why this was unreliable. Among 33 such comments in a 100-comment training audit:

| Sentiment based on text | Count |
| --- | ---: |
| Negative | 21 |
| Positive | 7 |
| Neutral | 1 |
| Unclear | 4 |

Increasing class weights did not sufficiently solve the problem:

| Group-based OOF metric | A1 | A2: balanced weights |
| --- | ---: | ---: |
| Neutral F1 | 0.0000 | 0.0822 |
| Positive recall | 0.9098 | 0.6112 |
| Macro-F1 | 0.4998 | 0.5006 |

During manual review, I changed the labels of 92 retained training records and excluded 43 records: 20 judged neutral and 23 unclear. This reduced the training pool from 2,520 to 2,477 records. Original comments and star ratings were preserved.

B1 uses **2,477 training records in 1,298 text groups**: 530 negative and 1,947 positive records. A text group contains comments with the same token sequence after preprocessing. Groups stay together during data splitting.

Some training labels still come from stars. Genuine neutral and unclear examples were kept outside binary training.

See [the labeling guide](labeling_guide.md) for the manual sentiment rules and examples.

## SQLite and data pipeline

1. [`db.py`](db.py) creates `data/processed/reviews.sqlite3`. [`ingest.py`](ingest.py) reads the CSV with pandas and loads the Zoom reviews into `reviews`, preserving the original comments and stars. The excluded review stays in the table with an exclusion reason.
2. [`explore.py`](explore.py) saves fastText language candidates in `language_predictions`. The manual review scripts save language, short text and letter-free decisions in separate tables linked by `review_id`.
3. [`prepare_training_data.py`](prepare_training_data.py) uses SQL `LEFT JOIN`s to combine reviews, language predictions and manual decisions. It selects training candidates and groups equivalent text so copies stay together. [`query_reviews.py`](query_reviews.py) shows how I checked counts and inspected the stored decisions.
4. [`select_evaluation_data.py`](select_evaluation_data.py) exports comments for manual evaluation and records their group assignments. After I reviewed the training labels, [`train_B1.py`](train_B1.py) used the revised training CSV and checked that development and test groups were excluded.

The training workflow uses local data and intermediate files. You can run the CSV demo below using the model already included in the repository.

## Decision thresholds

| Positive probability | Decision |
| --- | --- |
| Below 0.50 | `negative` |
| 0.50 to below 0.65 | `needs_review` |
| At least 0.65 | `positive` |

I chose this band by inspecting OOF probabilities from the first seed. I then repeated grouped five-fold cross-validation with two more seeds without changing it. This checked sensitivity to the fold split; it was not independent validation of the chosen thresholds.

Across three cross-validation seeds (equal weight per text group), automatic coverage ranged from **78.74% to 79.28%**, accuracy among automatic decisions from **90.51% to 91.13%**, and negative coverage from **76.06% to 76.83%**.

## Final evaluation

I used the development set for a behaviour check, then froze the model, thresholds and file hashes before evaluating 100 final test comments.

Cross-validation used training labels, some derived from stars. The final test used manual labels, so their accuracies are not directly comparable.

For the 86 positive/negative test comments:

| Manual label | Correct | Incorrect | Needs review |
| --- | ---: | ---: | ---: |
| Negative: 51 | 41 | 2 | 8 |
| Positive: 35 | 28 | 1 | 6 |

- Automatic coverage: **72/86 = 83.72%**
- Accuracy among automatic decisions: **69/72 = 95.83%**
- Approximate 95% Wilson interval for that accuracy: **88.45%–98.57%**
- Macro-F1 among automatic decisions: **0.9569**
- Negatives correctly identified automatically: **41/51 = 80.39%**

Across all 100 comments, 81 received automatic decisions. Nine were neutral or unclear and are excluded from the binary accuracy above. Only **2 of 6 neutral** and **3 of 8 unclear** comments were sent for review.

The test contains distinct text groups and is small. Its results do not establish performance on a live stream of reviews. The model and thresholds were not changed after the final test.

## Running predictions

Developed with Python 3.13 and scikit-learn 1.9.1.

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py examples\demo_input.csv --output data\processed\demo_predictions.csv
```

On Linux or macOS:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python predict.py examples/demo_input.csv --output data/processed/demo_predictions.csv
```

The demo uses the included `models/sentiment_B1.joblib` and `config/B1_frozen.json`. The application verifies the model and tokenizer hashes. Git checks out `text_features.py` with the CRLF line endings recorded when the model was frozen.

Input must be a UTF-8 CSV with a `content` column. `review_id` and `score` are optional.

```csv
review_id;content;score
example_1;Good app;5
example_2;I cannot join the meeting.;1
example_3;Good app;1
```

If the output file already exists, the application does not overwrite it; use a different output name.

Original columns are preserved. The output adds:

| Column | Meaning |
| --- | --- |
| `decision` | `positive`, `negative` or `needs_review` |
| `p_positive` | Model probability score; not a correctness guarantee |
| `input_status` | `empty`, `no_known_features` or `not_flagged` |
| `star_text_disagreement` | Opposing star group and text decision |

Blank inputs receive `needs_review` with no probability. Inputs with no learned feature matches also receive `needs_review`; their model score is retained for diagnosis. After the final test, I added an input check that sends comments with no known TF-IDF features to `needs_review`. A later check found that none of the 100 final test comments triggered this rule, so it does not change the reported test decisions. The model and thresholds remain unchanged.

Star disagreement is `true` for positive text predictions with 1–2 stars or negative predictions with 4–5 stars. It is `false` when these agree, and empty for three stars, missing ratings or `needs_review`.

## Limitations and learning

The model can struggle with mixed feedback, feature requests and positive expressions containing negative words, such as “no complaints.” It has no separate neutral class or prediction-time language filter. `not_flagged` does not guarantee understandable input.

I checked CSV processing with ten example rows, including emojis, blank text and conflicting stars. This was an application check, not another accuracy evaluation.

`DECISIONS.md` records my choices and findings. I used AI assistance for explanations, code drafting and debugging, while manually reviewing selected comments, making the project decisions and running the scripts locally.

## License

The original project code and documentation are available under the [MIT License](LICENSE). The source dataset and the fitted B1 model artifact are outside the scope of this license.
