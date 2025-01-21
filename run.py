from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("loan_prediction_model.pkl")

@app.route('/')
def home():
    return "Loan Prediction Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.get_json()

        # Extract features (ensure the order matches training features)
        features = np.array([data['Gender'], data['Married'], data['Education'],
                             data['Self_Employed'], data['ApplicantIncome'],
                             data['CoapplicantIncome'], data['LoanAmount'],
                             data['Loan_Amount_Term'], data['Credit_History'],
                             data['Property_Area']]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Return result as JSON
        return jsonify({
            'loan_approval': bool(prediction[0]),
            'message': "Loan approved" if prediction[0] else "Loan denied"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
