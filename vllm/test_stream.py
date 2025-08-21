import requests
import sseclient

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy-key"
}

json_data = {
    "model": "facebook/opt-125m",
    "prompt": "What's the capital of Canada?",
    "stream": True,
    "max_tokens": 32,
    "temperature": 0.7
}

response = requests.post(
    "http://localhost:8000/v1/completions",
    headers=headers,
    json=json_data,
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)