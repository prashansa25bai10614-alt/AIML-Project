# Fake Product Review Detection using Text Classification (Basic NLP)
# Requirements: pandas, scikit-learn
# Dataset: CSV file with columns 'review' (text) and 'label' (1=fake, 0=real)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load your dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['review'], df['label']

# 2. Preprocess and split data
def preprocess_and_split(reviews, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 3. Vectorize text (Bag of Words)
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

# 4. Train a simple classifier
def train_classifier(X_train_vec, y_train):
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_vec, y_train)
    return clf

# 5. Evaluate the model
def evaluate_model(clf, X_test_vec, y_test):
    y_pred = clf.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

# 6. Predict new reviews
def predict_review(review_text, vectorizer, clf):
    review_vec = vectorizer.transform([review_text])
    prediction = clf.predict(review_vec)[0]
    return 'Fake' if prediction == 1 else 'Genuine'

# Example usage
if __name__ == "__main__":
    # Replace 'reviews.csv' with your dataset path
    reviews, labels = load_data('reviews.csv')
    X_train, X_test, y_train, y_test = preprocess_and_split(reviews, labels)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    clf = train_classifier(X_train_vec, y_train)
    evaluate_model(clf, X_test_vec, y_test)

    # Test on a new review
    test_review = "This product is amazing! Best purchase ever!"
    print(f"Review: {test_review}")
    print("Prediction:", predict_review(test_review, vectorizer, clf))
