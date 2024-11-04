from flask import Flask, request, jsonify
import pickle

# Load the model and DictVectorizer
with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    X = dv.transform([client_data])
    prediction = model.predict_proba(X)[0, 1]
    return jsonify({'subscription_probability': round(prediction,3)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
