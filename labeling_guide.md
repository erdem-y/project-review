# Sentiment labeling guide

This guide documents the rules I used to review Zoom comments for sentiment. The examples below are invented.

Read the comment first and choose a label based on what it says about the app. Keep the original text and stars. Compare the label with the stars separately and use the notes field to explain difficult cases.

| Label | When to use it | Example |
| --- | --- | --- |
| `positive` | The user mainly expresses approval, satisfaction or praise. | “The app is easy to use. 👍” |
| `negative` | The user mainly expresses dissatisfaction or reports an unresolved problem. | “I can't join the meeting.” |
| `neutral` | The comment is understandable and expresses little or no clear sentiment. This includes a simple acknowledgement or a suggestion with no clear praise or complaint. | “Ok” or “A button to copy the meeting link would be useful.” |
| `unclear` | The comment gives too little understandable information to judge sentiment. | A random string, a number without context or a comment I cannot interpret. |

## Cases that need judgment

- **Mixed feedback.** Record `mixed` in `notes` when a comment contains praise and a complaint. Use the dominant sentiment as the main label. A suggestion can be secondary to an overall positive experience, as in “The calls have been great so far, though switching cameras could be easier.” If the text expresses no clear overall direction, use `neutral` for an understandable comment.
- **Emojis.** Treat a clear reaction such as `👍👌` as positive. Read combinations as a whole. An ambiguous symbol or a bare flag can stay `unclear`.
- **Star and text disagreement.** A one-star comment saying “Good app” has positive text sentiment. Three stars alone are insufficient for a neutral text label. Some training labels used stars as a starting point, while manually reviewed labels were based on the comment.
- **Equivalent comments.** Apply the same approved training label to records in the same text group, defined by their token sequence. Keep each group together when splitting the data.

## How the labels are used

The manual development and final test sets retain all four labels. B1 trains on `positive` and `negative`, with neutral and unclear cases reported separately during evaluation. `needs_review` is an application decision to defer a prediction. It is separate from the human sentiment labels.
