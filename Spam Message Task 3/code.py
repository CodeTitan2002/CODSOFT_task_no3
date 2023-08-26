import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('spam.csv',encoding='latin-1')
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['v2'])

X_train, X_test, y_train, y_test = train_test_split(X, data['v1'], test_size=0.2, random_state=42)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)

all_predictions = naive_bayes.predict(X)
data['Predicted'] = all_predictions

ordered_columns = ['v2', 'v1', 'Predicted']
data = data[ordered_columns]

data.to_csv('spam_with_predictions.csv', index=False)
print("Predictions saved to 'spam_with_predictions.csv'")
