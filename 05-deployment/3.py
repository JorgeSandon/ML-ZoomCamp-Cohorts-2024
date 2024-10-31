import pickle

# Load DictVectorizer and model
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

# Client data
client = {"job": "management", "duration": 400, "poutcome": "success"}

# Transform and predict
X = dv.transform([client])
proba = model.predict_proba(X)[0, 1]

print(f"Probability of subscription: {proba:.3f}")
