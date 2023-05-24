import argparse
import json

import requests
import threading
import time

def main(args: argparse.Namespace):
    prompt = [f"Tell me a story with more than {''.join([str(i+1)] * 5)} words"
              for i in range(args.n_thread)]

    headers = {"User-Agent": "CacheFlow Benchmark Client"}
    ploads = [{
        "prompt": prompt[i],
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    } for i in range(len(prompt))]

    def send_request(results, i):
        response = requests.post(args.api_url, headers=headers,
                                 json=ploads[i], stream=True)
        results[i] = response

    # use args.n_threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_thread
    for i in range(args.n_thread):
        t = threading.Thread(target=send_request, args=(results, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"Time (POST): {time.time() - tik} s")
    n_words = 0

    # if streaming:
    for i, response in enumerate(results):
        k = list(response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"))
        response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
        n_words += len(response_new_words.split(" ")) - len(prompt[i].split(" "))

    time_seconds = time.time() - tik
    print(f"Time (total): {time_seconds} to finish, n threads: {args.n_thread}, "
          f"throughput: {n_words / time_seconds} words/s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, default="http://localhost:8001")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--n-thread", type=int, default=2)
    args = parser.parse_args()

    main(args)