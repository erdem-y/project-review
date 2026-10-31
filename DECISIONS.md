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

## 005 — First sentiment model A1 and evaluation before A2

Before starting model training, I chose word-based TF-IDF with single words and two-word combinations, followed by logistic regression. I wanted to start with a simple approach that would help me understand which words and expressions influenced the predictions. I decided to try character-based features later and compare the results.

I used a custom tokenizer to keep emojis and expressions such as “can't” together. Two-word combinations also allowed the model to use expressions such as “not good”. The sentiment model was trained on my project data without using a pretrained sentiment model.

From the 2,650 candidates, I reserved 30 comments for development and 100 for the final test. I manually labelled these comments based on their content. I kept their text groups out of training, leaving 2,520 training records: 462 negative, 104 neutral and 1,954 positive.

Two development comments were marked as unclear, so I evaluated the model on the remaining 28. I kept the final test separate and did not generate predictions for it.

I called the first model A1. I started without class weighting so that I could use its results as a reference for later experiments.

For the first evaluation, I used five-fold cross-validation while keeping each text group together. In each fold, TF-IDF was learned only from the training part. This prevented the model from learning the vocabulary from the comments used for validation.

The average cross-validation macro-F1 was 0.5366. On the manually labelled development comments, the model correctly classified 23 out of 28 comments, giving an accuracy of 82.14% and a macro-F1 of 0.5644. However, it did not predict any comment as neutral.

I did not consider the development result enough to judge the model because it contained only 28 usable comments, including just two neutral comments. I also kept in mind that cross-validation used the existing training labels, many of which came from star ratings, while development used my manual labels.

Before moving to A2, I checked the tokenizer and examined the results more closely. All ten tokenizer checks passed. Some unusual tokens came from spelling mistakes or unclear source comments rather than a confirmed tokenizer problem. This also showed that a comment predicted as English was not necessarily understandable or suitable for sentiment training.

I then collected out-of-fold predictions, meaning that each training comment was predicted by a model that had not trained on its text group. I measured the results both by individual records and by giving each text group equal weight. This helped me see how repeated comments affected the evaluation.

This check revealed a problem with the first split. In one validation fold, all 426 positive records belonged to a single text group. The record counts were reasonably balanced, but the number of different positive groups was not.

To make the A1 and A2 comparison more consistent, I created new folds by distributing the unique text groups according to their sentiment labels. I kept all copies of a group together and saved the folds for reuse. I repeated the A1 evaluation without changing its model settings or training data.

With the new folds, A1 achieved a pooled out-of-fold macro-F1 of 0.5346 by record and 0.4998 by text group. A simple baseline that always predicted the most common training class achieved 0.2912 and 0.2473 respectively.

For the group-based evaluation, negative F1 was 0.6675, positive F1 was 0.8321 and neutral F1 was 0.0000. Across all 2,520 out-of-fold predictions, the model predicted 413 comments as negative and 2,107 as positive, with no neutral predictions.

These results showed that A1 performed better than the majority baseline, but it still failed to identify neutral comments. Its record-based accuracy of 86.47% did not show this weakness clearly, so I decided not to rely on accuracy alone. The small neutral class and the difference between a three-star rating and neutral wording may contribute to the problem, but I have not established their individual effects.

Before running A2, I decided to change only class weighting to “balanced” and reuse the same folds. I will prefer A2 if its group-based pooled macro-F1 improves by at least 0.02, its neutral F1 increases, its positive recall drops by no more than 0.05, and its group-based macro-F1 improves in at least four of the five folds.

I set these practical comparison rules before seeing A2's results. After this comparison, I plan to review a separate sample of 100 training comments for data quality. The final test remains unused.

006 — Testing balanced class weights and reviewing training labels
For A2, I changed only classifier__class_weight to "balanced". This gave more weight to individual records from smaller classes and less weight to records from larger classes. I wanted to see whether this would help the model learn neutral comments, which A1 had not predicted at all.

I kept the same training data, text features and saved validation folds so that I could compare the effect of class weighting without changing other settings. The final test remained unused.

A2 did not meet the comparison rules I had set before running it. Its pooled group-based macro-F1 increased only from 0.4998 to 0.5006, below my required improvement of 0.02. It improved in two of the five folds instead of at least four.

The model became better at finding negative comments, but it also assigned the negative label to more comments that belonged to other classes. Group-based negative recall increased from 0.6311 to 0.9244, while negative precision decreased from 0.7082 to 0.5417. Positive recall dropped from 0.9098 to 0.6112, much more than the maximum drop of 0.05 I had accepted.

Neutral F1 increased from 0.0000 to 0.0822 in the group-based evaluation. However, this improvement was too small to justify the loss in positive recall.

Of the 98 neutral groups, only six were correctly predicted as neutral. The model predicted 74 as negative and 18 as positive. Of its 48 neutral predictions, only six were correct. Another 28 were actually labelled positive and 14 negative, giving neutral precision of 0.1250.

These results made me question what the model was learning from the neutral training labels. Many neutral-labelled comments were predicted as negative. One possible explanation was that three-star comments contained complaints rather than clearly neutral wording. Increasing their weight would not necessarily solve that problem.

I had already been cautious about the neutral class because it had few training records. Among the 130 comments I manually labelled for development and testing, only eight were neutral, around 6.2%. However, these comments were selected using class quotas, so I did not treat this percentage as the neutral share of all reviews.

Before deciding whether to remove neutral from training, I wanted to understand how much of the problem came from the training labels. I reviewed another 100 comments from the training pool: 34 with negative labels, 33 with neutral labels and 33 with positive labels.

The clearest finding was in the neutral group. Of the 33 comments originally labelled neutral, I labelled 21 as negative and seven as positive. Only one remained neutral, while four were unclear.

This showed that star-based neutral labels often did not match the sentiment expressed in this sample. It supported my concern that treating every three-star review as textually neutral was not working well.

I decided to review the remaining neutral training groups before preparing the next training version. There were 98 neutral groups in total, and the sample covered 33 of them, leaving 65 to review.

My next plan is to use the manual decisions to prepare a positive–negative training set. Genuinely neutral and unclear comments will remain stored separately and will not be forced into either class. I will preserve the original comments and ratings.

A1 remains the preferred reference from the A1–A2 comparison. After completing the label review, I plan to train a new binary baseline and then investigate when the model should leave a prediction for manual review.