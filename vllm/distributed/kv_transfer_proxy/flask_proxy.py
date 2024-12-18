from flask import Flask, request, Response
import httpx
import time
from datetime import datetime

app = Flask(__name__)

# URLs for the two vLLM processes
VLLM_1_URL = "http://localhost:8100/v1/completions"
VLLM_2_URL = "http://localhost:8200/v1/completions"


def send_request_to_vllm(vllm_url, req_data):
    """Send request to a vLLM process and return the response."""
    with httpx.Client(timeout=None) as client:
        response = client.post(vllm_url, json=req_data)
        response.raise_for_status()


def stream_vllm_response(vllm_url, req_data):
    """Stream response from a vLLM process."""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", vllm_url, json=req_data) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                yield chunk


@app.route("/v1/completions", methods=["POST"])
def proxy_request():
    req_data = request.get_json()

    # Log initial request data
    print(len(req_data['prompt']), req_data['prompt'][0:10],
          "----received request :",
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Send request to vLLM-1
    send_request_to_vllm(VLLM_1_URL, req_data)

    # Log after first response
    print(len(req_data['prompt']), req_data['prompt'][0:10],
          "----response 1 :",
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Stream response from vLLM-2 back to the client
    def generate_stream():
        for chunk in stream_vllm_response(VLLM_2_URL, req_data):
            yield chunk

    # Log before sending final response
    print(len(req_data['prompt']), req_data['prompt'][0:10],
          "----response 2 :",
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Return streaming response using Flask's Response object
    return Response(generate_stream(), content_type="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
