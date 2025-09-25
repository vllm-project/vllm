import argparse
import os
import shutil
import time
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

import requests
from datasets import load_dataset
from PIL import Image


def to_data_url(image_path: str, fmt: str = "JPEG") -> str:
    """Read image from disk and convert to base64 data URL (robust for vLLM OpenAI server)."""
    with Image.open(image_path).convert("RGB") as img:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"


def build_messages(question_text: str, image_url: str) -> List[Dict[str, Any]]:
    """Mimic the SGLang prompt: one user message containing image + question text."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": question_text},
            ],
        }
    ]


def chat_once(
    api_base: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    seed: int = -1,
    timeout: int = 120,
    api_key: str = "",
) -> Tuple[str, int]:
    """
    Send one /chat/completions request to vLLM OpenAI-compatible server.
    Return (answer_text, completion_tokens).
    """
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if top_k is not None and top_k >= 0:
        payload["top_k"] = top_k
    if seed is not None and seed >= 0:
        payload["seed"] = seed

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return text, completion_tokens


def main(args):
    # Apply deterministic overrides if requested
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    seed = args.seed

    if args.deterministic:
        # for deterministic test
        temperature = 0.0
        top_p = 1.0
        top_k = -1
        repetition_penalty = 1.0
        if seed is None or seed < 0:
            seed = 42

    # Prepare cache dirs (same structure as the SGLang script)
    cache_dir = os.path.join(".cache", "mmstar")
    image_dir = os.path.join(cache_dir, "images")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    print(f"Created temporary image directory: {cache_dir}")

    # Read data (MMStar val)
    dataset = load_dataset("Lin-Chen/MMStar")["val"]

    # Build requests
    requests_payload = []
    for idx, q in enumerate(dataset):
        if idx >= args.num_questions:
            break
        # Save image to expected nested path under cache_dir (e.g., images/2.jpg)
        rel_path = q["meta_info"]["image_path"]  # e.g., "images/2.jpg"
        image_path = os.path.join(cache_dir, rel_path)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        q["image"].convert("RGB").save(image_path, "JPEG")

        # Strip options from question text, same as SGL script
        question_text = q["question"].split("Options:", 1)[0].strip()

        # Use data URL so we don't depend on --allowed-local-media-path
        img_url = to_data_url(image_path, fmt="JPEG")

        messages = build_messages(question_text, img_url)
        requests_payload.append(messages)

    # Fire requests (parallel similar to num_threads in SGLang run_batch)
    tic = time.perf_counter()
    completion_tokens_sum = 0
    answers: List[str] = [""] * len(requests_payload)

    def submit_one(messages):
        return chat_once(
            args.api_base,
            args.model,
            messages,
            args.max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
            args.timeout,
            args.api_key,
        )

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(submit_one, messages): i for i, messages in enumerate(requests_payload)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                text, ctk = fut.result()
                print(f"text={text},ctk={ctk}")
                answers[i] = text
                completion_tokens_sum += ctk
            except Exception as e:
                answers[i] = f"[ERROR] {e}"

    latency = time.perf_counter() - tic

    # Compute throughput (tokens/s) — matches SGL's "Output throughput"
    output_throughput = completion_tokens_sum / latency if latency > 0 else 0.0

    # Accept length: SGLang reports spec_verify_ct; not available via OpenAI API -> set to 1.0
    # accept_length = 1.0

    # Print results (same fields as SGL script)
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Cleanup
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted temporary directory: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Keep SGL-like knobs
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--parallel", type=int, default=8, help="Number of concurrent requests")
    # vLLM OpenAI-compatible endpoint args
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8080/v1", help="vLLM OpenAI-compatible base URL")
    # If you didn't set --served-model-name when launching vLLM, set this to your served name or path
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Served model name or path recognized by vLLM")
    parser.add_argument("--api-key", type=str, default="", help="Bearer token if your server requires auth")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)

    # Sampling / determinism controls
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1, help="-1 to disable; non-negative to enable")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1, help="Fixed RNG seed; <0 means unset")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic-like settings: temp=0, top_p=1, top_k=-1, rep_penalty=1, seed=42 if unset")

    args = parser.parse_args()
    main(args)