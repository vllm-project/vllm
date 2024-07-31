from quart import Quart, request, jsonify, Response
import httpx

app = Quart(__name__)

async def forward_request(url, data):
    async with httpx.AsyncClient() as client:
        async with client.stream('POST', url, json=data) as response:
            if response.status_code == 200:
                # Check if the response is streaming
                if 'transfer-encoding' in response.headers and response.headers['transfer-encoding'] == 'chunked':
                    # Stream the response
                    async def stream_response():
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    return Response(stream_response(), status=200, content_type=response.headers.get('content-type'))
                else:
                    # Return the full response
                    response_data = await response.aread()
                    return Response(response_data, status=200, content_type=response.headers.get('content-type'))
            else:
                error_data = await response.aread()
                return jsonify({'error': error_data.decode()}), response.status_code

@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    # Get the original request data
    original_request_data = await request.get_json()

    # Modify the max_tokens to 1 for the request to port 8100
    modified_request_data_8100 = original_request_data.copy()
    modified_request_data_8100['max_tokens'] = 1

    # Forward the request to port 8100
    response_8100 = await forward_request('http://localhost:8100/v1/completions', modified_request_data_8100)

    if response_8100.status_code == 200:
        # If the request to port 8100 is successful, forward the original request to port 8200
        response_8200 = await forward_request('http://localhost:8200/v1/completions', original_request_data)

        if response_8200.status_code == 200:
            return response_8200
        else:
            return jsonify({'error': 'Failed to get response from port 8200'}), response_8200.status_code
    else:
        return jsonify({'error': 'Failed to get response from port 8100'}), response_8100.status_code

if __name__ == '__main__':
    app.run(port=8000)
