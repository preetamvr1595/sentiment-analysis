# IMDB Sentiment Analysis — Complete Data Analysis + Machine Learning + Deep Learning (50,000 Reviews)

End-to-end **sentiment analysis in Python** on the **IMDB 50K movie-review dataset**: a full exploratory data analysis first, then text preprocessing, classical machine learning (TF-IDF + Logistic Regression / SVM / Naive Bayes) and deep learning (PyTorch: embeddings, TextCNN, BiGRU). Every notebook is fully executed — read the outputs right on GitHub.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![NLP](https://img.shields.io/badge/NLP-sentiment--analysis-purple) ![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red) ![scikit--learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)

## Results at a glance

| Model | Test accuracy |
|---|---|
| **TF-IDF (1–2-gram) + LinearSVC** | **91.72%** |
| TF-IDF (1–2-gram) + Logistic Regression | 91.60% |
| Multinomial Naive Bayes (TF-IDF 1–2-gram) | 89.30% |
| Averaging-embedding network (PyTorch) | 89.13% |
| BiGRU (PyTorch) | 89.11% |
| TextCNN (PyTorch) | 87.56% |
| Text-statistics only (length, punctuation…) | ≈ 62% |

Honest finding: on 40k reviews, a tuned **linear model on TF-IDF bigrams beat small neural networks** — and trains in seconds. Pre-trained transformers (not run here, they need a GPU) are the next step.

## Notebooks (run in order)

| # | Notebook | Stage | Cells |
|---|---|---|---|
| 01 | `01_dataset_overview_and_quality.ipynb` | EDA — shape, dtypes, missing values, duplicates, class balance, noise, leakage check | 59 |
| 02 | `02_eda_text_statistics.ipynb` | EDA — length, word/sentence stats, punctuation, capitals, correlations, significance tests | 73 |
| 03 | `03_eda_vocabulary_ngrams.ipynb` | EDA — vocabulary, Zipf/Heaps laws, word clouds, log-odds words, n-grams, negation | 64 |
| 04 | `04_text_preprocessing.ipynb` | Cleaning pipeline, measured effect of each step | 50 |
| 05 | `05_classical_ml_models.ipynb` | 12 classical models, CV, GridSearch, ROC, error analysis, explanations | 71 |
| 06 | `06_deep_learning_pytorch.ipynb` | PyTorch: AvgEmbed, TextCNN, BiGRU, comparison, ensembling | 61 |

**378 cells in total.**

## Key insights from the analysis
- 50,000 reviews → 49,582 after removing duplicates; perfectly balanced classes.
- ≈ 11.4M tokens, 119k-word vocabulary; the top 20k words cover 97% of the text.
- Length / punctuation alone give only ~62% accuracy — the signal is in the **words**.
- Heavy cleaning barely helps; **bigrams** give the biggest lift (≈ +1 point). Removing stop-words or stemming slightly *hurt*.
- Negation (*not good*) flips meaning — handled by n-grams, missed by order-blind models.

## Dataset
IMDB Large Movie Review Dataset (Maas et al., 2011), 50,000 labelled reviews as one CSV (`data/IMDB_Dataset.csv`, 66 MB).

## Quick start
```bash
git clone https://github.com/preetamvr1595/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
jupyter notebook          # run notebooks 01 → 06 in order
```
Notebooks 01–04 create `data/imdb_dedup.csv`, `imdb_with_stats.csv` and `imdb_clean.csv` (git-ignored; some exceed GitHub's file-size limit).

## FAQ
**Which dataset is used for IMDB sentiment analysis?** The 50K labelled IMDB movie reviews, 25k positive and 25k negative.
**What accuracy can I expect?** About 91–92% with TF-IDF bigrams and a linear model; ~94% with fine-tuned transformers.
**Do I need a GPU?** No — everything here runs on a CPU (the BiGRU takes ~23 minutes).

## Part of the Data Science Mastery series
[numpy-mastery](https://github.com/preetamvr1595/numpy-mastery) · [pandas](https://github.com/preetamvr1595/pandas) · [matplotlib](https://github.com/preetamvr1595/matplotlib) · [seaborn](https://github.com/preetamvr1595/seaborn)

If this helped you, please ⭐ star the repo.


*Keywords: sentiment analysis, IMDB, NLP, text classification, TF-IDF, scikit-learn, PyTorch, LSTM, GRU, CNN, exploratory data analysis, movie reviews*
