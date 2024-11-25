from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the model and the scaler
with open('model/dtc_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Main route (Page with the form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to process the form and make the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal']),
        ]
        
        # Scale the input data
        features_scaled = scaler.transform([features])

        # Make the prediction
        prediction = model.predict(features_scaled)

        # Return the result to the user
        result = "Positive for heart disease" if prediction[0] == 1 else "Negative for heart disease"
        
        # Use jsonify to send the prediction as JSON
        return jsonify(prediction_result=result)

    except Exception as e:
        return f"Error: {e}"

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


