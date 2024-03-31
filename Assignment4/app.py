from flask import Flask, request, jsonify
from score import score
import joblib

model = joblib.load("Support_Vector_Machine.pkl")

app = Flask(__name__)

@app.route('/score', methods=['POST', 'GET'])
def get_score():
    data = request.json
    text = data['text']
    threshold = float(data['threshold'])
    
    prediction, propensity = score(text, model, threshold)
    
    response = {
        'prediction': int(prediction),
        'propensity': propensity
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port = 5000)  # Run the Flask app