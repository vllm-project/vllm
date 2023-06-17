import argparse
import json
import threading
import time

import requests


def main(args: argparse.Namespace):
    prompts = [f"Tell me a story with more than {''.join([str(i+1)] * 5)} words"
              for i in range(args.n_threads)]

    api_url = f"http://{args.host}:{args.port}/generate"
    headers = {"User-Agent": "vLLM Benchmark Client"}
    ploads = [{
        "prompt": p,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    } for p in prompts]

    def send_request(results, i):
        response = requests.post(api_url, headers=headers, json=ploads[i],
                                 stream=True)
        results[i] = response

    # use args.n_threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_threads
    for i in range(args.n_threads):
        t = threading.Thread(target=send_request, args=(results, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"Time (POST): {time.time() - tik} s")
    n_words = 0

    for i, response in enumerate(results):
        k = list(response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"))
        response_new_words = json.loads(k[-2].decode("utf-8"))["text"][0]
        n_words += len(response_new_words.split(" ")) - len(prompts[i].split(" "))

    time_seconds = time.time() - tik
    print(f"Time (total): {time_seconds:.3f}s to finish, n_threads: {args.n_threads}, "
          f"throughput: {n_words / time_seconds} words/s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-threads", type=int, default=128)
    args = parser.parse_args()

    main(args)
