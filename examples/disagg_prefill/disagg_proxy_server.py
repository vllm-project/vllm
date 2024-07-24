import http.server
import socketserver
import requests
import json
import argparse

class ProxyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.prefill_port = kwargs.pop('prefill_port', 8100)
        self.decode_port = kwargs.pop('decode_port', 8200)
        super().__init__(*args, **kwargs)

    def do_POST(self):
        # Read the content length to get the data size
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Parse the JSON payload
        data = json.loads(post_data)
        
        # Change the max_tokens to 1 for the request to prefill_port
        data_prefill = data.copy()
        data_prefill["max_tokens"] = 1
        post_data_prefill = json.dumps(data_prefill)
        
        # Forward the request to prefill_port with modified max_tokens
        response_prefill = requests.post(f"http://localhost:{self.prefill_port}/v1/completions", 
                                         headers={"Content-Type": "application/json"},
                                         data=post_data_prefill)
        
        # Check if the response from prefill_port is successful
        if response_prefill.status_code == 200:
            # Forward the original request to decode_port
            response_decode = requests.post(f"http://localhost:{self.decode_port}/v1/completions", 
                                            headers={"Content-Type": "application/json"},
                                            data=post_data)
            
            # Send the response back to the client
            self.send_response(response_decode.status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response_decode.content)
        else:
            # Send an error response back to the client
            self.send_response(response_prefill.status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response_prefill.content)

def run_server(port_8000, prefill_port, decode_port):
    handler = lambda *args, **kwargs: ProxyHTTPRequestHandler(*args, prefill_port=prefill_port, decode_port=decode_port, **kwargs)
    with socketserver.TCPServer(("", port_8000), handler) as httpd:
        print(f"Serving at port {port_8000}")
        httpd.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proxy server")
    parser.add_argument('prefill_port', type=int, help='Port to forward the first request to (with max_tokens=1)')
    parser.add_argument('decode_port', type=int, help='Port to forward the second request to')
    args = parser.parse_args()
    
    run_server(8000, args.prefill_port, args.decode_port)
