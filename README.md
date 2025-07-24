# Sentiment-Analysis
COMPANY: CODTECH IT SOLUTIONS

NAME: MADHURA MAHAJAN

INTERN ID: CT06DZ81

DOMAIN: DATA ANALYTICS

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

**DESCRIPTION OF THE PROJECT :**
📘 Project Overview: Sentiment Analysis with NLTK and Machine Learning

This project is a complete **Sentiment Analysis pipeline** designed to classify text (such as tweets, reviews, or comments) into three categories:

* **Positive**
* **Negative**
* **Neutral**

It leverages the power of **Natural Language Processing (NLP)** and **Supervised Machine Learning** to understand the emotional tone of text data.

🔍 Project Goals

* Clean and preprocess raw text using **NLTK** tools.
* Convert text into numerical data using **TF-IDF** and **Count Vectorization**.
* Train and evaluate multiple machine learning models for accuracy.
* Visualize insights such as word clouds, sentiment distribution, and important features.

🧱 Core Components of the Project

1. **Library Integration**

The project uses:

* **NLTK**: For stopword removal, tokenization, and lemmatization.
* **Scikit-learn**: For machine learning models and data transformation.
* **Pandas/NumPy**: For data handling and manipulation.
* **Seaborn/Matplotlib/WordCloud**: For data visualization.

2. **Data Handling**

* If available, a CSV file containing labeled text data is loaded.
* If not, a **small built-in sample dataset** is used for demonstration.
* The data must have two columns: the actual text and its sentiment label (e.g., "I love this product" → Positive).

3. **Text Preprocessing**

To prepare raw text for analysis:

* **Lowercasing**: Makes the analysis case-insensitive.
* **Removing noise**: Deletes links, usernames, punctuation, hashtags, and numbers.
* **Tokenization**: Splits the text into individual words.
* **Stopword removal**: Filters out common but meaningless words like “is”, “the”, etc.
* **Lemmatization**: Converts words to their base forms (e.g., “running” → “run”).
  
  This step ensures that only meaningful and uniform words remain for analysis.

4. **Data Visualization**

* **Sentiment Distribution Chart**: A bar chart showing how many positive, negative, and neutral entries are in the dataset.
* **Word Clouds**: Visually show the most frequent words used in each sentiment category. Larger words appear more frequently.

These charts help us understand the nature of the data and common vocabulary associated with each emotion.

5. **Text Vectorization**

To use machine learning on text, words must be converted to numbers:

* **TF-IDF Vectorizer**: Gives more weight to unique words and less to common ones.
* **Count Vectorizer**: Simple count of how many times each word appears.

Both methods are tested to compare their effect on model performance.

6. **Model Training and Evaluation**

The text data is split into **training** and **testing** sets.

Then, the following four machine learning models are trained:

1. **Naive Bayes** – Fast and effective for text.
2. **Logistic Regression** – Great for binary/multiclass classification.
3. **Support Vector Machine (SVM)** – Works well for margin-based separation.
4. **Random Forest** – A powerful tree-based ensemble model.

Each model is evaluated using:

* **Accuracy score** – Overall percentage of correct predictions.
* **Confusion matrix** – Shows actual vs predicted classes.
* **Classification report** – Detailed precision, recall, and F1-score metrics.

This helps identify which model performs best and why.

✅ What You Learn From This Project

* The complete lifecycle of a text classification task.
* Practical use of NLTK for text cleaning and preprocessing.
* How vectorization changes model performance.
* Differences in performance between several machine learning algorithms.
* How to interpret confusion matrices and classification reports.



