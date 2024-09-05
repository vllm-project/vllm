"""An example showing how to use vLLM to serve VLMs.

Launch the vLLM server with the following command:

(single image inference with Llava)
vllm serve llava-hf/llava-1.5-7b-hf --chat-template template_llava.jinja

(multi-image inference with Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --max-model-len 4096 \
    --trust-remote-code --limit-mm-per-prompt image=2
"""
import base64

import requests
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Single-image input inference
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

## Use image url in the payload
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What’s in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            },
        ],
    }],
    model=model,
    max_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print("Chat completion output:", result)


## Use base64 encoded image in the payload
def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""

    with requests.get(image_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


image_base64 = encode_image_base64_from_url(image_url=image_url)
chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What’s in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ],
    }],
    model=model,
    max_tokens=64,
)

result = chat_completion_from_base64.choices[0].message.content
print(f"Chat completion output:{result}")

# Multi-image input inference
image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What are the animals in these images?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url_duck
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url_lion
                },
            },
        ],
    }],
    model=model,
    max_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print("Chat completion output:", result)
