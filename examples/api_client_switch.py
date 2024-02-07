import argparse
import json
from typing import Iterable, List
import pdb
import requests

def post_http_request(modeltype: str,
                      api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "modeltype": modeltype,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--modeltype", type=str, default="Baichuan2-13B")
    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/switch"
    response = post_http_request(args.modeltype, api_url)
    data = json.loads(response.content)
    print(data)