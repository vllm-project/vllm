import os

#os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import subprocess
import time

import requests
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def run_server(args):
    serve_cmd = [
        "vllm",
        "serve",
        args.model,
        "--max_num_seqs",
        str(args.batchsize),
        "--max_num_batched_tokens",
        str(args.batchsize * args.max_model_len),
        "--disable-uvicorn-access-log",
    ]

    if args.enforce_eager:
        serve_cmd.append("--enforce_eager")

    proc = subprocess.Popen(
        serve_cmd,
    )

    return proc


def wait():
    while True:
        try:
            api_url = "http://localhost:8000/v1/embeddings"
            prompt = {
                "model": args.model,
                "input": "vLLM is great!",
            }
            response = requests.post(api_url, json=prompt)
            if response.status_code == 200:
                break
        except Exception:
            pass

        time.sleep(5)


def _benchmark(args):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    prompt = "你" * (args.input_len[0] - 2)
    def worker(prompt):
        api_url = "http://localhost:8000/v1/embeddings"
        prompt = {
            "model": args.model,
            "input": prompt,
            "encoding_format": "bytes",
            "embed_dtype": "fp8_e4m3"
        }
        start = time.perf_counter()
        #print("start: ", start)
        response = requests.post(api_url, json=prompt)
        assert response.status_code == 200
        len(response.content)
        end = time.perf_counter()
        #print("end: ", end)
        e2e = end - start
        print(e2e * 1000)

    for i in range(10):
        print("=" * 80)
        worker(prompt)


def benchmark(args):
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(_benchmark, args)
        f.result()


def run(args):
    for batchsize in args.batchsizes:
        try:
            args.batchsize = batchsize
            proc = run_server(args)
            wait()

            benchmark(args)
        finally:
            proc.terminate()


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsizes = [1]
    args.input_len = [32]
    args.n_clients_list = [1]

    args.enforce_eager = False

    run(args)
