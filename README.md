# 🧠 Text Analytics Research Projects (2025)

This repository presents three independent text analytics research projects focused on **climate-related financial disclosures** and **social media entity recognition**. Each project investigates real-world NLP challenges using a combination of classical algorithms and deep learning models.

---

## 📁 Project Overview

| Project | Title | Domain |
|--------|-------|--------|
| **1** | Climate Sentiment Classification | ESG, Sentiment Analysis, Transfer Learning |
| **2** | Topic Modeling of Climate Disclosures | Unsupervised Learning, NLP |
| **3** | Named Entity Recognition on Twitter | Social Media, Sequence Tagging |

---

## 🌱 Project 1 — Climate Sentiment Classification

This project focuses on classifying climate-related corporate disclosures into **Risk**, **Opportunity**, or **Neutral** categories.

### 🔧 Methods

- **Models**:
  - Multinomial Naïve Bayes
  - Feedforward Neural Network (FFNN)
  - TinyBERT (lightweight transformer model)
- **Techniques**:
  - CountVectorizer vs. TF-IDF
  - N-gram tuning
  - Error analysis on ambiguous and indirect texts

### 📊 Highlights

| Model        | Validation Accuracy | Test Accuracy |
|--------------|---------------------|---------------|
| Naïve Bayes  | 0.755               | 0.790         |
| FFNN         | 0.550               | 0.490         |
| TinyBERT     | 0.750               | 0.785         |

- **Naïve Bayes** with unigrams was most effective due to domain-specific keyword cues.
- **TinyBERT** generalized well with low-resource training, thanks to pretrained contextual embeddings.
- **FFNN** underperformed due to limited representation power without pretrained input.

---

## 🌍 Project 2 — Topic Modeling for Climate Disclosures

This project explores semantic themes in ESG reports using modern topic modeling techniques.

### 🛠 Methods

- **Primary Approach**: BERTopic (BERT + UMAP + HDBSCAN)
- **Baselines**:
  - LDA (Latent Dirichlet Allocation)
  - KMeans with MiniLM embeddings
- **Evaluation**:
  - Topic Coherence
  - Diversity
  - Label Purity & Entropy

### 🔬 Key Findings

| Method                  | Coherence | Diversity | Silhouette |
|-------------------------|-----------|-----------|------------|
| LDA                    | 0.4363    | –         | –          |
| KMeans + MiniLM        | –         | –         | 0.0195     |
| **BERTopic**           | **0.5821**| **0.7450**| –          |

- Fine-tuning `min_topic_size` revealed trade-offs between granularity and interpretability.
- Clear label-topic alignment shown via label purity (>90% in some topics).
- BERTopic demonstrated best ability to uncover **climate risk vs. opportunity** themes.

---

## 🐦 Project 3 — Named Entity Recognition on Twitter

This project develops a domain-specific sequence tagger for noisy Twitter data using the **Broad Twitter Corpus (BTC)**.

### 🧠 Model

- **Architecture**: BERTweet (Transformer pretrained on Twitter)
- **Tag Format**: BIO (Begin–Inside–Outside)
- **Token Alignment**: Custom alignment for subwords using `is_split_into_words=True`

### 🧪 Results

| Model     | Precision | Recall | F1 Score | Loss    |
|-----------|-----------|--------|----------|---------|
| BERTweet | 0.8074    | 0.7975 | **0.8004** | 0.1348  |

- Domain-specific pretraining led to **58-point F1 gain** over vanilla BERT.
- Highest recognition for PERSON > LOCATION > ORGANIZATION.
- Errors included partial multi-token spans and ambiguity between entities.

---

## 🛠️ Technologies Used

- Python 3.10+
- Libraries:
  - `scikit-learn`, `transformers`, `datasets`
  - `bertopic`, `umap-learn`, `hdbscan`
  - `pandas`, `seaborn`, `matplotlib`
- Jupyter Notebook, HuggingFace, and BERTweet

---

## 📂 Repository Structure

```plaintext
text-analytics-projects/
├── project1_climate_sentiment/
│   ├── sentiment_classification_nb_bert_ffnn.ipynb
│   └── report_summary_q1_q2.1.md
├── project2_climate_topics/
│   ├── topic_modeling_bertopic_vs_lda.ipynb
│   └── results_topic_eval.pdf
├── project3_twitter_ner/
│   ├── ner_training_bertweet.ipynb
│   ├── eval_f1_by_entity.ipynb
│   └── examples_errors_analysis.txt
└── README.md
