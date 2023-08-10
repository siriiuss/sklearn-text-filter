import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


model = joblib.load("text_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

while True:
    input_text = input("Enter data:\t")


    input_vectorized = vectorizer.transform([input_text])
    prediction = model.predict(input_vectorized)
    predicted_label = "objectionable" if prediction[0] == 1 else "acceptable"

    print(f"Filtered as {predicted_label}")

    user_feedback = input("True? (yes/no): ")

    if user_feedback.lower() == 'yes':
        continue
    elif user_feedback.lower() == 'no':
        correct_label = input("Please enter true situation (objectionable/acceptable): ")
        correct_label_encoded = 1 if correct_label == "objectionable" else 0
        new_data = [input_text]
        new_labels = [correct_label_encoded]

        X_new_vectorized = vectorizer.transform(new_data)
        model.partial_fit(X_new_vectorized, new_labels)
        joblib.dump(model, 'text_classifier_model.joblib')
        print("Updated.")
    else:
        print("Invalid feedback.")
