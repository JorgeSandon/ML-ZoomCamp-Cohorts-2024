import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("data/heart.csv")

# Split the dataset into X (features) and y (target)
X = df.drop('target', axis=1)
y = df['target']

# Scale the X data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Train the decision tree model
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dtc.predict(X_test)

# Print the evaluation metrics
print(classification_report(y_test, y_pred))

# Perform cross-validation and print the results
scores = cross_val_score(dtc, X_train, y_train, cv=10)
print("After cross-validation with cv = 10, the average accuracy is", scores.mean(), "and the standard deviation of cross-validation is", scores.std())

# Save the trained model using pickle
with open('model/dtc_model.pkl', 'wb') as file:
    pickle.dump(dtc, file)

with open('model/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

