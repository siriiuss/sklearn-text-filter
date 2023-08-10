from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


objectionable_texts = ["This text objectionable.", "This text contains objectionable arguments."]
acceptable_texts = ["This text is acceptable.", "This text contains acceptable arguments."]
texts = objectionable_texts + acceptable_texts
tags = [1] * len(objectionable_texts) + [0] * len(acceptable_texts)


X_train, X_test, y_train, y_test = train_test_split(texts, tags, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


joblib.dump(model, 'text_classifier_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
