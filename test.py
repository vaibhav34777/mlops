import requests
url = "http://0.0.0.0:8000/predict"

payload = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 3.2
}

response = requests.post(url, json=payload)

# Print response from API
print(response.status_code)
print(f"Duration: {response.json()}")
