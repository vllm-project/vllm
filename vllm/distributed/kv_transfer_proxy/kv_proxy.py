from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import asyncio

app = FastAPI()

# URLs for the two vLLM processes
# VLLM_1_URL = "http://10.192.18.145:8000/v1/completions"
# VLLM_2_URL = "http://10.192.24.218:8000/v1/completions"

VLLM_1_URL = "http://localhost:8000/v1/completions"
VLLM_2_URL = "http://localhost:8001/v1/completions"


async def send_request_to_vllm(vllm_url, req_data):
    """Send request to a vLLM process and return the response."""
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(vllm_url, json=req_data)
        response.raise_for_status()
        return


async def stream_vllm_response(vllm_url, req_data):
    """Asynchronously streams the response from a vLLM process."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", vllm_url, json=req_data) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():

                # print("streaming chunk", chunk)
                yield chunk


@app.post("/v1/completions")
async def proxy_request(request: Request):
    # Extract the incoming request JSON data
    import time
    from datetime import datetime
    req_data = await request.json()

    print(len(req_data['prompt']), req_data['prompt'][0:10],
          "----received request :",
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Send request to vLLM-1 and wait for its response
    # Response from vLLM-1 is not sent back to the user
    response1 = await send_request_to_vllm(VLLM_1_URL, req_data)

    print(len(req_data['prompt']), req_data['prompt'][0:10],
          "----response 1 :",
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Optionally, you can process response1 and modify req_data if needed
    # For now, we'll proceed with the original req_data

    # Stream response from vLLM-2 back to the client
    async def generate_stream():
        chunk_count = 0
        async for chunk in stream_vllm_response(VLLM_2_URL, req_data):
            if chunk_count == 0:
                print(
                    len(req_data['prompt']), req_data['prompt'][0:10],
                    "----First chunk sent at:",
                    datetime.fromtimestamp(
                        time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
            chunk_count += 1
            yield chunk

        print(
            len(req_data['prompt']), req_data['prompt'][0:10],
            "----Last chunk sent at:",
            datetime.fromtimestamp(
                time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
        print(len(req_data['prompt']), req_data['prompt'][0:10],
              f"Total chunks sent: {chunk_count}")

    print(len(req_data['prompt']), req_data['prompt'][0:10],
          "----response 2 :",
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Use StreamingResponse to stream the response back to the client
    return StreamingResponse(generate_stream(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
