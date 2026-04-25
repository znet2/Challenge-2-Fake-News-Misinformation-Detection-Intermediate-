================================================================
FAKE NEWS DETECTION — NeuroLogic '26 | Challenge 2
================================================================

TEAM / AUTHOR : [Your Name / Team Name]
DATE          : April 25, 2026
COMPETITION   : NeuroLogic '26 — Department of AIML, GGITS

----------------------------------------------------------------
RESULTS
----------------------------------------------------------------
Validation Accuracy : 99.96%
Precision (Fake)    : 1.00
Recall    (Fake)    : 1.00
F1-Score  (Fake)    : 1.00
Precision (Real)    : 1.00
Recall    (Real)    : 1.00
F1-Score  (Real)    : 1.00

Validation set size : 2,670 samples (15% stratified split)
  - Fake (FALSE)    : 1,162
  - Real (TRUE)     : 1,508

----------------------------------------------------------------
APPROACH & METHODOLOGY
----------------------------------------------------------------
Task:
  Binary classification — predict whether a news article is
  Real (TRUE) or Fake (FALSE) based on its title and text.

Model:
  DistilBERT (distilbert-base-uncased) fine-tuned for
  sequence classification via Hugging Face Transformers.

  DistilBERT is a distilled version of BERT that retains ~97%
  of BERT's language understanding while being 40% smaller and
  60% faster — ideal for hackathon time constraints.

Input Construction:
  input = title + " [SEP] " + text[:512]

  Combining title and text gives the model both the headline
  signal and the article body context. Text is truncated to
  512 characters before tokenization (max_length=256 tokens).

Training Details:
  - Epochs            : 3
  - Batch size        : 16 (train), 32 (eval)
  - Optimizer         : AdamW (default in Trainer)
  - Weight decay      : 0.01
  - Warmup steps      : 100
  - Mixed precision   : fp16 (on GPU)
  - Best model        : loaded by validation accuracy
  - Train/Val split   : 85% / 15% stratified

Label Encoding:
  TRUE  (Real) → 1
  FALSE (Fake) → 0

----------------------------------------------------------------
FILE STRUCTURE
----------------------------------------------------------------
  fake_news_detection.ipynb   — Main Colab notebook (runnable)
  README.txt                  — This file
  requirements.txt            — Python dependencies
  no_label.csv                — Predictions on test set
  fakenews_with_labels.csv    — Training data (provided)
  FakeNews_no_labels.csv      — Test data (provided)

----------------------------------------------------------------
INSTRUCTIONS TO REPRODUCE
----------------------------------------------------------------
1. Open Google Colab: https://colab.research.google.com
2. Upload fake_news_detection.ipynb
3. Set Runtime → Change runtime type → T4 GPU
4. Upload both CSV files to the Colab file panel:
     fakenews_with_labels.csv
     FakeNews_no_labels.csv
5. Run all cells in order (Runtime → Run all)
6. submission.csv (no_label.csv) will be downloaded automatically

Expected total runtime: ~8–12 minutes on T4 GPU

----------------------------------------------------------------
ENVIRONMENT
----------------------------------------------------------------
Platform  : Google Colab (Python 3.12)
GPU       : NVIDIA T4 (free tier)
See requirements.txt for full package versions.
================================================================
