# NLP Web-based Data Application for Clothing Recommendation

---

## Project Overview

This project presents a comprehensive **Natural Language Processing solution for Clothing Recommendation Classification**. The system involves developing an end-to-end NLP pipeline to classify clothing reviews and predict whether customers will recommend products based on their review text, using advanced text preprocessing, feature extraction, and machine learning techniques on approximately 1,960 clothing reviews.

---

## Access to Code 🔒

The source code for this project is **private** and available upon request.

If you're an **employer** or **recruiter** and would like to review the code, please **request access via email** at **ayanhashmi205@yahoo.com**.

---

## Installation 📦

#### Clone this repo

```bash
git clone https://github.com/ayan9870/nlp-clothing-recommendation.git
cd nlp-clothing-recommendation
```

#### Launch notebooks

```bash
jupyter notebook Module1.ipynb
jupyter notebook Module2_3.ipynb
```
---

## Dependencies ⚡

* Python 3.7+
* pandas
* numpy
* scikit-learn
* nltk
* gensim (for word embeddings)
* matplotlib
* seaborn
* jupyter

---

## Features 📋

* **Text Preprocessing Pipeline**: Comprehensive text cleaning with tokenization, stopword removal, and frequency filtering
* **Multiple Feature Representations**: Bag-of-words, TF-IDF weighted, and word embedding models
* **Binary Classification**: Predict product recommendations (recommended vs not-recommended)
* **Cross-validation**: 5-fold cross-validation for robust model evaluation
* **Language Model Comparison**: Evaluation of different embedding models (FastText, Word2Vec, GloVe)
* **Multi-feature Analysis**: Comparison of title-only, review-only, and combined features
* **Automated Vocabulary Generation**: Custom vocabulary creation with proper indexing

---

## Algorithm 🔍

### Natural Language Processing Pipeline

#### Module 1: Text Preprocessing
* **Tokenization**: Using regex pattern `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`
* **Normalization**: Lowercase conversion and length filtering
* **Stopword Removal**: Using custom English stopword list
* **Frequency Filtering**: Remove single-occurrence and top-20 frequent words
* **Vocabulary Building**: Alphabetically sorted vocabulary with integer indexing

#### Module 2: Feature Representation
* **Bag-of-Words**: Count vector representation based on generated vocabulary
* **Word Embeddings**: Weighted (TF-IDF) and unweighted vector representations
* **Multiple Models**: FastText, GoogleNews300, Word2Vec, or GloVe embeddings

#### Module 3: Classification Models
* **Logistic Regression**: Primary classification model with sklearn
* **Cross-validation**: 5-fold validation for robust performance measurement
* **Feature Comparison**: Title vs Review vs Combined feature analysis

---

## Dataset 📊

* **Source**: Women's E-Commerce Clothing Reviews (modified for course)
* **Size**: ~1,960 reviews
* **Features**: 
  - Title: Review title
  - Review: Detailed product review
  - Recommended: Binary label (0=not-recommended, 1=recommended)

---

## Results Summary 📈

| Experiment | Focus | Methodology |
|------------|-------|-------------|
| Language Model Comparison | Best embedding model | 5-fold cross-validation comparison |
| Feature Information Analysis | Title vs Review vs Combined | Performance impact of different text features |
| Classification Performance | Recommendation prediction | Binary classification accuracy |

---

## Usage 🚀

1. **Text Preprocessing**: Run `Module1.ipynb` to clean and process the review data
2. **Feature Extraction**: Execute `Module2_3.ipynb` to generate different feature representations
3. **Model Training**: Train and evaluate classification models using the same notebook
4. **Analysis**: Compare different language models and feature combinations

---

## Author ✨

| <a href="https://github.com/ayan9870" target="_blank">**Muhammad Ayan Hashmi**</a> |
|:--:|
| ![Ayan Hashmi](https://github.com/ayan9870.png?size=100) |
| [`github.com/ayan9870`](https://github.com/ayan9870) |

---

## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
