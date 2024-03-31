import joblib
import requests
import subprocess
import time
from score import score

# Load the best model saved during experiments
best_model = joblib.load("Support_Vector_Machine.pkl")

def test_score():
    # Smoke test: Check if the function produces some output without crashing
    text = "Test text"
    threshold = 0.5
    prediction, propensity_score = score(text, best_model, threshold)
    assert prediction is not None
    assert propensity_score is not None
    print('Smoke Test: Success')

    # Format test: Check if the output formats/types are as expected
    assert isinstance(prediction, bool)
    assert isinstance(propensity_score, float)
    print('Format Test: Success')

    # Prediction value test: Check if the prediction value is 0 or 1
    assert prediction in [0, 1]
    print('Prediction Value Test: Success')

    # Propensity score range test: Check if the propensity score is between 0 and 1
    assert 0 <= propensity_score <= 1
    print('Propensity Score Test: Success')

    # Threshold tests
    # If the threshold is set to 0, the prediction should always be 1
    prediction, _ = score(text, best_model, threshold=0)
    assert prediction == 1
    print('0 Threshold Test: Success')

    # If the threshold is set to 1, the prediction should always be 0
    prediction, _ = score(text, best_model, threshold=1)
    assert prediction == 0
    print('1 Threshold Test: Success')

    # Obvious spam input text test: Prediction should be 1
    spam_text = "Congratulations! You have won a lottery of 1 MILLION dollars! Claim your reward now by clicking the link below!"
    prediction, propensity_score = score(spam_text, best_model, threshold=0.5)
    assert prediction == 1
    print('Spam Test: Success')

    # Obvious non-spam input text test: Prediction should be 0
    non_spam_text = "Hello, please send me your letter of resignation by tonight."
    prediction, propensity_score = score(non_spam_text, best_model, threshold=0.5)
    assert prediction == 0
    print('Non-Spam Test: Success')

    print('All test cases passed successfully.')

# Integration test function
def test_flask():
    process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE)

    time.sleep(2)

    payload = {"text":"Congratulations! You have been selected to go on a free trip to Hawaii! Don't miss out on this wondeful opportunity! This is a limited time offer so hurry up and fill the form quickly!", "threshold":0.5}
    response = requests.post('http://127.0.0.1:5000/score', json=payload)
    

    print("Status code:", response.status_code)
    print("Response body:", response.text)   
    
    data = response.json()
    
    assert 'prediction' in data
    assert 'propensity' in data

    process.terminate()
    