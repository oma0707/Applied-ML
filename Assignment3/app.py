from flask import Flask, request, jsonify
from score import score
import joblib

# Load the best model
model = joblib.load("Support_Vector_Machine.pkl")

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def get_score():
    data = request.json
    text = data['text']
    threshold = float(data['threshold'])
    
    # Score the text using the best model
    prediction, propensity = score(text, model, threshold)
    
    response = {
        'prediction': int(prediction),
        'propensity': propensity
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app