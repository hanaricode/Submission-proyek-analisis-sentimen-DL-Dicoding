# 📊 Sentiment Analysis of Shopee App Reviews

![Jupyter-Notebook](https://img.shields.io/badge/Jupyter_Notebook-Lang-FFE873)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-orange)
![NLP](https://img.shields.io/badge/NLP-Indonesian-FF0000)
![Deep-Learning](https://img.shields.io/badge/Deep_Learning-Coming_Soon-87CEEB)

## 📝 Description
This project is an implementation of sentiment analysis on **Shopee** application reviews from the **Google Play Store** using Natural Language Processing (NLP) and Machine Learning techniques. The data was obtained through independent scraping of **12,000 reviews** in Indonesian language. The sentiment labeling was done using a **Lexicon-Based** approach (InSet Lexicon) to ensure labels are based on actual text analysis rather than star ratings.

---

## 📁 File Structure
```
├── Notebook_Scraping_Analisis_Sentimen.ipynb       # Data scraping notebook
├── Pelatihan_Model_Analisis_Sentimen.ipynb         # Main training notebook + inference
├── notebook_scraping_analisis_sentimen.py          # Scraping code in .py format
├── pelatihan_model_analisis_sentimen.py            # Training code in .py format
├── ulasan_aplikasi_shopee.csv                      # Raw scraped dataset
├── ulasan_shopee_labeled.csv                       # Labeled dataset
├── model_sentimen.pkl                              # Saved best model (SVM)
├── vectorizer.pkl                                  # Saved TF-IDF vectorizer
└── requirements.txt                                # Library dependencies
```

---

## 🔄 Workflow
| Step | Description |
|------|-------------|
| 1. **Data Scraping** | Scraping 12,000 Shopee app reviews from Google Play Store |
| 2. **Data Preparation** | Remove null values and duplicates |
| 3. **Text Preprocessing** | Cleaning, casefolding, slang word normalization, tokenizing, stopword filtering, stemming |
| 4. **Data Labeling** | Lexicon-Based sentiment labeling using InSet Lexicon |
| 5. **Data Visualization** | Distribution charts, word clouds, polarity score histogram |
| 6. **Balancing Data** | Equalize the number of data per class |
| 7. **Feature Extraction** | TF-IDF (Term Frequency - Inverse Document Frequency) |
| 8. **Model Training** | 3 different training schemes |
| 9. **Evaluation** | Accuracy, precision, recall, F1-score |
| 10. **Inference** | Predict sentiment of new text |
---

## 📌 Dataset
| Info | Detail |
|------|--------|
| **Source** | Google Play Store (Shopee App) |
| **Total Data** | 12,000 reviews |
| **After Preprocessing** | 11,648 reviews |
| **After Balancing** | 4,836 reviews |
| **Language** | Indonesian |
| **Labeling Method** | Lexicon-Based (InSet Lexicon) |
| **Classes** | Positive, Negative, Neutral |

### Label Distribution (Before Balancing)
| Class | Total |
|-------|-------|
| Positive | ~6,200 |
| Negative | ~3,800 |
| Neutral | ~1,600 |

### Label Distribution (After Balancing)
| Class | Total |
|-------|-------|
| Positive | 1,612 |
| Negative | 1,612 |
| Neutral | 1,612 |

---

## ⚙️ Preprocessing Steps
| Step | Method | Description |
|------|--------|-------------|
| Cleaning | Regex | Remove URLs, hashtags, mentions, numbers, special characters |
| Casefolding | Python `.lower()` | Convert all text to lowercase |
| Slang Normalization | Custom dictionary | Normalize informal/slang words |
| Tokenizing | NLTK `word_tokenize` | Split sentences into word tokens |
| Stopword Removal | NLTK + Sastrawi | Remove unimportant words |
| Stemming | PySastrawi | Reduce words to their base form |

---

## 🤖 Training Models
| Scheme | Algorithm | Feature Extraction | Data Split | Train Accuracy | Test Accuracy |
|--------|-----------|-------------------|------------|----------------|---------------|
| 1 | SVM (Support Vector Machine) | TF-IDF | 80/20 | 99.25% | 89.26% |
| 2 | Random Forest | TF-IDF | 80/20 | 99.84% | 88.53% |
| 3 | Logistic Regression | TF-IDF | 70/30 | 97.96% | 85.67% |

> ✅ All schemes meet the minimum accuracy requirement of **85%**

### Combination Differences
- **Combination 1** — Different algorithms (SVM vs Random Forest vs Logistic Regression)
- **Combination 2** — Different data split (80/20 vs 70/30)

---

## 📊 Best Model Results
**Model: SVM + TF-IDF + Split 80/20**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.98 | 0.98 | 0.98 | 322 |
| Neutral | 0.98 | 0.98 | 0.98 | 323 |
| Positive | 0.98 | 0.98 | 0.98 | 323 |
| **Accuracy** | | | **0.98** | **968** |

---

## 🔍 Inference Example
```python
Input  : "aplikasi shopee sangat bagus, fitur lengkap dan mudah digunakan"
Output : POSITIF

Input  : "aplikasi sering error, lemot, dan sangat mengecewakan sekali"
Output : NEGATIF

Input  : "aplikasi shopee cukup lumayan, kadang bagus kadang tidak"
Output : NETRAL
```
⚠️ Note: The inference results may not always be accurate for every sentence. This is due to the limitations of the InSet Lexicon dictionary used for labeling, where some words may have ambiguous or incorrect sentiment values (e.g., a positive word being assigned a negative score). The model predicts based on patterns learned from the labeled data, which was generated using this lexicon. For more accurate results, consider using a Deep Learning approach such as LSTM, GRU, or BERT in future development(comingsoon).

---

## 📦 Libraries Used
| Library | Version | Purpose |
|---------|---------|---------|
| `google-play-scraper` | 1.2.7 | Data scraping from Google Play Store |
| `Sastrawi` | latest | Indonesian stemming & stopword removal |
| `nltk` | latest | Tokenization & stopwords |
| `scikit-learn` | latest | ML algorithms & TF-IDF feature extraction |
| `gensim` | latest | Word2Vec |
| `wordcloud` | latest | Word cloud visualization |
| `pandas` | latest | Data processing |
| `numpy` | latest | Array operations |
| `matplotlib` | latest | Data visualization |
| `seaborn` | latest | Statistical visualization |

---

## ⚙️ How to Run

### 1. Clone this repository
```bash
git clone https://github.com/hanaricode/Submission-proyek-analisis-sentimen-DL-Dicoding.git
cd Submission-proyek-analisis-sentimen-DL-Dicoding
```

### 2. Install all dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the scraping notebook first
```bash
jupyter notebook Notebook_Scraping_Analisis_Sentimen.ipynb
```

### 4. Run the main training notebook
```bash
jupyter notebook Pelatihan_Model_Analisis_Sentimen.ipynb
```

---

## 👤 Author & 📄 License
- **Name** : Hanari
- © 2026 Hanari. All Rights Reserved.
