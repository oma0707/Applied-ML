from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    # Define the preprocessing pipeline
    preprocess_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000)),
    ])
    train = pd.read_csv("train.csv")
    train_X, train_y = train.transformed_text, train.Spam
    
    preprocess_pipeline.fit(train_X)

    # Preprocess the input text
    preprocessed_text = preprocess_pipeline.transform([text])

    # Perform prediction
    predicted_proba = model.predict_proba(preprocessed_text)[0]
    propensity_score = predicted_proba[1]  # Assuming the positive class is at index 1

    # Apply threshold for binary classification
    prediction = bool(propensity_score >= threshold)

    return prediction, float(propensity_score)
