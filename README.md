# 📝 Text Analytics Projects: Climate Sentiment & Social Media NER

This repository presents two independent text analytics research projects focused on **climate-related financial disclosures** and **social media entity recognition**. The work explores both classical and modern natural language processing (NLP) techniques, combining traditional models with pretrained language models.

---

## 📁 Project Overview

| Project | Title | Domain |
|--------|-------|--------|
| **1** | Climate Sentiment Classification & Topic Modeling | ESG, NLP, Transfer Learning |
| **2** | Named Entity Recognition on Twitter Data | Social Media, Sequence Tagging |

---

## 🌍 Project 1 — Climate Sentiment Classification & Topic Discovery

This project analyzes corporate climate disclosures to classify text into **climate risks**, **opportunities**, or **neutral statements**, and explores topic discovery methods to identify thematic structures in ESG-related narratives.

### 🔧 Methods Used

- **Text Classification Models**:
  - Multinomial Naïve Bayes (CountVectorizer-based)
  - Feedforward Neural Network (FFNN with random embeddings)
  - TinyBERT (Transformer model for transfer learning)

- **Modifications & Observations**:
  - Unigram features improve Naïve Bayes performance by reducing sparsity.
  - FFNN performs poorly due to lack of contextual embedding.
  - TinyBERT demonstrates strong generalization and benefits from pretrained knowledge.

- **Topic Modeling Techniques**:
  - **BERTopic** (BERT + UMAP + HDBSCAN)
  - **LDA** (Latent Dirichlet Allocation)
  - **KMeans with MiniLM Embeddings**

- **Evaluation Metrics**:
  - Classification Accuracy, Misclassification Analysis
  - Topic Coherence, Diversity, Entropy, Purity

### 📊 Key Results

- **Naïve Bayes**: Best classifier with 79% test accuracy  
- **BERTopic**: Outperformed LDA and KMeans with highest coherence (0.58) and diversity (0.74)  
- **Granularity Tuning**: Smaller topic sizes improve interpretability and semantic richness  
- **Class Mapping**: Topic-to-label alignment evaluated using entropy and purity measures

---

## 🐦 Project 2 — Named Entity Recognition on Twitter

This project builds a domain-adapted sequence tagging system to extract **people**, **organizations**, and **locations** from informal and noisy Twitter data.

### 🔧 Model Setup

- **Architecture**: Transformer-based model using [`vinai/BERTweet`](https://huggingface.co/vinai/bertweet-base)
- **Tokenizer Handling**: Custom token-label alignment to handle subword tokenization
- **Tagging Scheme**: BIO (Begin, Inside, Outside) format for entity spans

### 💡 Features and Design

- Subword tokenization for robustness to misspellings and abbreviations
- Domain-specific pretraining (Twitter corpus) improves contextual modeling
- Linear classification head with cross-entropy loss

### 📈 Evaluation

- **Dataset**: Broad Twitter Corpus (BTC)
- **Metrics**: Precision, Recall, F1-score (macro-averaged across entity types)
- **Performance**:
  - BERTweet achieved **F1 = 0.80**, outperforming BERT baseline by over 58 points
  - Recognition best for **PER** > **LOC** > **ORG**
- **Error Patterns**:
  - Partial tagging of multi-token entities
  - Confusions between PERSON and ORGANIZATION types
  - Missed entities due to spelling variation or abbreviations

---

## 🛠️ Technologies Used

- Python 3.10+
- Libraries: `scikit-learn`, `transformers`, `datasets`, `pandas`, `matplotlib`, `seaborn`, `bertopic`, `nltk`
- HuggingFace Transformers & Datasets
- BERTopic for topic modeling
- Jupyter Notebooks for experimentation

---

## 📂 Repository Structure

```plaintext
text-analytics-projects/
├── project1_climate_sentiment/
│   ├── sentiment_classification.ipynb
│   ├── topic_modeling_bertopic.ipynb
│   ├── data/
│   │   └── climate_disclosures.csv
│   └── outputs/
│       └── figures, tables, topic maps
├── project2_ner_twitter/
│   ├── ner_training_bertweet.ipynb
│   ├── eval_ner_results.ipynb
│   └── btc_dataset/
└── README.md
