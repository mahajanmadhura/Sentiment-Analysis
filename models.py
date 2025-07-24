from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(X_train, X_test, y_train, y_test, vectorizer_type="tfidf"):
    print(f"\n=== Model Training with {vectorizer_type.upper()} ===")
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear'),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    results = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        results[name] = accuracy
    return results

def feature_importance(X_train, y_train, vectorizer):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_
    top_features = pd.DataFrame({'feature': feature_names, 'importance': importances}
                                ).sort_values('importance', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('Top 10 Important Features for Sentiment Analysis')
    plt.show()
    print("\nTop 10 Important Features:")
    print(top_features)
