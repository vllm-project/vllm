import requests

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model":
        "TIGER-Lab/VLM2Vec-Full",
        "messages": [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Represent the given image."
                },
            ],
        }],
        "encoding_format":
        "float",
    },
)
response.raise_for_status()
response_json = response.json()

print("Embedding output:", response_json["data"][0]["embedding"])
