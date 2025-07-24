import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

from nltk_utils import download_nltk_resources, clean_text
from data_utils import load_data
from visualization import visualize_data
from models import train_models, feature_importance

def main():
    download_nltk_resources()
    df = load_data()
    print("\nOriginal Data Sample:")
    print(df.head())
    df['cleaned_text'] = df['text'].apply(clean_text)
    print("\nCleaned Text Sample:")
    print(df[['text', 'cleaned_text']].head())
    visualize_data(df)
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_num'] = df['sentiment'].map(sentiment_map)
    X = df['cleaned_text']
    y = df['sentiment_num']
    print("\n=== Feature Extraction ===")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    count_vectorizer = CountVectorizer(max_features=5000)
    X_count = count_vectorizer.fit_transform(X)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42)
    X_train_count, X_test_count, _, _ = train_test_split(
        X_count, y, test_size=0.2, random_state=42)
    train_models(X_train_tfidf, X_test_tfidf, y_train, y_test, "tfidf")
    train_models(X_train_count, X_test_count, y_train, y_test, "count")
    print("\n=== Feature Importance Analysis ===")
    feature_importance(X_train_tfidf, y_train, tfidf_vectorizer)

if __name__ == "__main__":
    main()
