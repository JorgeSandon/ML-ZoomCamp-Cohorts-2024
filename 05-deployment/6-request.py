import requests


url = "http://localhost:5000" 
client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client)
result = response.json()
print(result)
