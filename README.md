# Feedback Analysis

A Python application that classifies review sentiment and flags disagreements with star ratings.

**Current policy:** B1 uses the 0.59–0.65 decision band. In the latest external check, 859 of 1,000 comments received automatic decisions. Against AI references, 742 of 815 automatic decisions on positive/negative comments matched (91.04%); 44 neutral/unclear comments also received automatic sentiment labels.

**Original frozen test:** with the earlier 0.50–0.65 band and manual labels, 69 of 72 automatic positive/negative decisions were correct. The two evaluations use different samples and reference labels.

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
| Below 0.59 | `negative` |
| 0.59 to below 0.65 | `needs_review` |
| At least 0.65 | `positive` |

The current application uses [`config/B1_inference.json`](config/B1_inference.json). The original 0.50–0.65 policy remains recorded in [`config/B1_frozen.json`](config/B1_frozen.json) for the historical final evaluation.

I originally chose the 0.50–0.65 band by inspecting OOF probabilities from the first seed. I then repeated grouped five-fold cross-validation with two more seeds without changing it. This checked sensitivity to the fold split; it was not independent validation of the chosen thresholds.

For that original band, across three cross-validation seeds (equal weight per text group), automatic coverage ranged from **78.74% to 79.28%**, accuracy among automatic decisions from **90.51% to 91.13%**, and negative coverage from **76.06% to 76.83%**.

## Original final evaluation (0.50–0.65)

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

The test contains distinct text groups and is small. Its results do not establish performance on a live stream of reviews. These results describe the original frozen policy. I later selected a new application band using external data, as explained below; the original test report and frozen configuration are preserved.

## External testing and threshold changes

After the final test, I wanted to see how the model would perform on a larger set of new comments. I collected 1,000 new Zoom reviews and used AI to label them based on their text, without using their star ratings or individual model predictions. Comments matching the original data by review ID or tokenised text group were excluded.

These AI labels were reference labels, not human-verified answers. The original final test used comments I labelled manually, so I kept this difference in mind when comparing the results.

In the first external test, the model automatically classified 741 comments and sent 259 for review. Among comments labelled positive or negative, it made 697 automatic decisions, with 638 matching the references. Accuracy was 91.54%, compared with 95.83% in the original final test. Automatic coverage also fell from 81.00% to 74.10%. The share of negative comments correctly identified fell from 80.39% to 60.47%, while the share of positive comments correctly identified stayed similar, increasing from 80.00% to 80.77%.

What bothered me was not only the lower scores, but also how many comments needed manual review. Since the purpose of my project was to analyse comments automatically, sending around a quarter of them back to a person felt too much. My goal was to reduce manual review while keeping mistakes at an acceptable level. I did not want to improve accuracy simply by avoiding difficult comments.

I explored these decision bands during the process:

`0.50–0.65 → 0.56–0.64 → 0.56–0.70 → 0.56–0.60 → 0.56–0.64 → 0.56–0.66 → 0.59–0.68 → 0.59–0.70 → 0.59–0.65`

During the external testing, the 0.56–0.66 band sent 202 comments for review. I manually examined their text, probabilities and review reasons. Many comments around 0.55–0.59 expressed negative experiences but were still being sent for review. This encouraged me to move the lower threshold to 0.59.

Some negative comments received positive probabilities above 0.50. This suggested that the model leaned towards positive predictions in some cases. I think the training distribution may partly explain this, because it contained considerably more positive than negative records. I did not separately measure how much of this behaviour came from that imbalance.

I also learned that 90 comments in that review list had no features recognised by the model. They all received approximately the same probability, 0.66437. This was the classifier's output for an empty feature vector, rather than a judgement based on understanding those comments. A separate input check already sent them for review regardless of the thresholds.

I settled on 0.59–0.65 and fixed it before collecting another 1,000 comments. The latest sample did not overlap the original data or any earlier external download by review ID or tokenised text group. AI labelled every comment before its predictions were examined. I did not adjust the band using this last sample, and the B1 model remained unchanged.

The external samples were collected from Google Play in newest-first order using en/us store settings. Older reviews were reached after excluding previous data. Previously seen text groups were excluded, so these are novel-text samples, not representative samples of the full live review stream. This does not guarantee that every comment is English. They are also not prospective tests on future reviews.

| Measure | Original final test | First external test | Latest external test |
| --- | ---: | ---: | ---: |
| Comments | 100 | 1,000 | 1,000 |
| Decision band | 0.50–0.65 | 0.50–0.65 | **0.59–0.65** |
| Reference labels | Manual | AI | AI |
| Negative references | 51 | 430 | 412 |
| Positive references | 35 | 468 | 520 |
| Neutral references | 6 | 44 | 44 |
| Unclear references | 8 | 58 | 24 |
| Automatic decisions | 81 | 741 | **859** |
| Automatic coverage | 81.00% | 74.10% | **85.90%** |
| Sent for review | 19 | 259 | **141** |
| Correct negative decisions | 41 | 260 | **326** |
| Positive comments incorrectly labelled negative | 1 | 15 | **35** |
| Correct positive decisions | 28 | 378 | **416** |
| Negative comments incorrectly labelled positive | 2 | 44 | **38** |
| Accepted binary accuracy | 95.83% — 69/72 | 91.54% — 638/697 | **91.04% — 742/815** |
| Accepted binary macro-F1 | 0.9569 | 0.9129 | **0.9093** |
| Negative precision, binary only | 97.62% | 94.55% | **90.30%** |
| Positive precision, binary only | 93.33% | 89.57% | **91.63%** |
| All negative references correctly identified | 80.39% | 60.47% | **79.13%** |
| All positive references correctly identified | 80.00% | 80.77% | **80.00%** |
| Neutral/unclear comments given an automatic sentiment | 9 | 44 | **44** |
| Exact reference agreement across all automatic decisions | 85.19% | 86.10% | **86.38%** |

Accepted binary accuracy, precision and F1 include only automatically classified comments whose reference label was positive or negative. They exclude review decisions and neutral/unclear references. The last row includes neutral/unclear references and counts automatic sentiment decisions on them as disagreements.

The latest test had this distribution:

| Reference label | Negative decision | Positive decision | Sent for review |
| --- | ---: | ---: | ---: |
| Negative | 326 | 38 | 48 |
| Positive | 35 | 416 | 69 |
| Neutral | 13 | 18 | 13 |
| Unclear | 6 | 7 | 11 |
| **Total** | **380** | **479** | **141** |

To isolate the threshold effect, I also compared the original policy with the selected policy on this same latest sample. This was a comparison against the existing baseline, not another search for a new band.

| Measure on the latest 1,000 comments | 0.50–0.65 | 0.59–0.65 |
| --- | ---: | ---: |
| Automatic decisions | 747 | **859** |
| Sent for review | 253 | **141** |
| Correct negative decisions | 245 | **326** |
| Incorrect negative decisions on positive comments | 12 | **35** |
| Accepted binary accuracy | 92.97% | **91.04%** |
| Negative precision, binary only | 95.33% | **90.30%** |
| All negative references correctly identified | 59.47% | **79.13%** |
| Neutral/unclear comments given an automatic sentiment | 36 | **44** |

Raising the lower threshold produced 81 additional correct negative decisions, but also 23 additional incorrect negative decisions and negative labels for eight additional neutral/unclear comments. Positive decisions stayed unchanged because the upper threshold remained at 0.65.

I was satisfied with this precision–recall trade-off. For my project, correctly identifying more complaints and sending fewer comments for review was more useful than keeping higher precision while leaving many understandable comments unclassified. I therefore selected **0.59–0.65 as the final application decision band**.

This does not mean the model is error-free. In the latest test, 44 of 68 neutral/unclear comments still received an automatic sentiment. The samples also have different class distributions, so the changes between tests cannot be attributed only to thresholds. The AI references may contain mistakes and were produced by the same AI annotator. The first external sample was later used during threshold exploration; the last sample was held out until the new band had been selected.

Aggregated counts and metric definitions are included in [`evaluation/B1_external_summary.json`](evaluation/B1_external_summary.json). The raw external comments and annotation files are not included in this repository.

Testing another app and completing the project
For my last evaluation, I wanted to check whether the model could also handle a larger dataset from another app. I tested it on 5,000 Microsoft Teams reviews using the same B1 model and the fixed 0.59–0.65 decision band. I did not retrain the model or change the thresholds for Teams.

AI labelled the comments before their predictions were examined. There were 3,837 different exact texts; identical copies received the same reference label. None of the review IDs appeared in the previous data. These labels are AI references, not human-verified answers.
Measure	Latest Zoom test	Microsoft Teams test
Comments	1,000	5,000
Automatic decisions	859	4,534
Automatic coverage	85.90%	90.68%
Sent for review	141 — 14.10%	466 — 9.32%
Correct negative decisions	326	1,362
Positive comments incorrectly labelled negative	35	144
Correct positive decisions	416	2,696
Negative comments incorrectly labelled positive	38	137
Accepted binary accuracy	91.04% — 742/815	93.52% — 4,058/4,339
Accepted binary macro-F1	0.9093	0.9285
Negative precision, binary only	90.30%	90.44%
Positive precision, binary only	91.63%	95.16%
All negative references correctly identified	79.13%	80.74%
All positive references correctly identified	80.00%	89.09%
Accuracy with equal weight per text group	91.02%	90.97%
Exact reference agreement across all automatic decisions	86.38%	89.50%


Accepted binary accuracy, precision and F1 include only automatic decisions on reference-positive/negative comments. Neutral, unclear and review cases are excluded. The last row includes automatic decisions on neutral/unclear references and counts them as disagreements.

Compared with the latest Zoom test, automatic coverage increased by 4.78 percentage points, while the review rate decreased by the same amount. Accepted binary accuracy increased by 2.48 percentage points, and positive precision increased by 3.53 percentage points. Negative precision stayed almost the same. The share of all negative comments correctly identified increased by 1.61 percentage points, while the corresponding positive share increased by 9.09 percentage points.

However, I also checked how repeated comments affected these results. Different apps can receive the same short comments, such as “Good” or “Nice”. In the Teams sample, 1,427 comments had token sequences also found in the original Zoom source data.

When every text group received equal weight, accuracy was 90.97%, slightly below the previous Zoom result of 91.02%. Among the remaining 3,573 Teams comments whose token groups were not found in the original Zoom source, automatic coverage was 87.80% and accepted binary accuracy was 90.69%. This showed that the higher record-based score was partly helped by familiar and repeated expressions, while performance on different wording remained similar.

The limitations also remained. Of 287 neutral or unclear references, 195 received an automatic sentiment label. Therefore, the 93.52% binary accuracy should not be presented as accuracy across every comment. Across all 4,534 automatic decisions, exact reference agreement was 89.50%.

This larger test from another app gave me more confidence in the version I had selected. I did not see a major performance drop, and some record-based results were higher. It does not prove that the model will perform equally well on every app, but it was enough for the practical goal of this project.

I consider B1 with the 0.59–0.65 decision band the best balance I reached in this project. It automatically handles most comments, identifies a useful share of complaints, and keeps manual review relatively small. I accept that some mistakes remain and have documented them separately.

I am satisfied with this version and consider the project complete.

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

The demo uses the included `models/sentiment_B1.joblib` and the current `config/B1_inference.json` policy with thresholds 0.59 and 0.65. `config/B1_frozen.json` preserves the original test policy. The application verifies the model and tokenizer hashes. Git checks out `text_features.py` with the CRLF line endings recorded when the model was frozen.

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

Blank inputs receive `needs_review` with no probability. Inputs with no learned feature matches also receive `needs_review`; their model score is retained for diagnosis. After the final test, I added an input check that sends comments with no known TF-IDF features to `needs_review`. A later check found that none of the 100 final test comments triggered this rule, so it does not change the reported test decisions. The model remains unchanged; the current 0.59–0.65 application policy is documented separately from the historical test policy.

Star disagreement is `true` for positive text predictions with 1–2 stars or negative predictions with 4–5 stars. It is `false` when these agree, and empty for three stars, missing ratings or `needs_review`.

## Limitations and learning

The model can struggle with mixed feedback, feature requests and positive expressions containing negative words, such as “no complaints.” It has no separate neutral class or prediction-time language filter. `not_flagged` does not guarantee understandable input.

I checked CSV processing with ten example rows, including emojis, blank text and conflicting stars. This was an application check, not another accuracy evaluation.

`DECISIONS.md` records my choices and findings. I used AI assistance for explanations, code drafting and debugging, while manually reviewing selected comments, making the project decisions and running the project scripts locally. AI also collected, labelled and evaluated the external samples, which are reported separately from the manual final test.

## License

The original project code and documentation are available under the [MIT License](LICENSE). The source dataset and the fitted B1 model artifact are outside the scope of this license.
