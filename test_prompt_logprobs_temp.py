"""
Minimal test: compare argmax token logprob at different temperatures.

Usage:
  1. Start a vLLM server:
     python -m vllm.entrypoints.openai.api_server \
         --model <your-model> --port 8000

  2. Run this script:
     python test_prompt_logprobs_temp.py [--model <model-name>] [--port 8000]

Expected: low temp → argmax logprob closer to 0 (more peaked),
          high temp → argmax logprob more negative (flatter distribution).
"""
import argparse
import requests


def get_prompt_logprobs(base_url: str, model: str, prompt: str,
                        prompt_logprobs_temperature: float | None = None):
    """Send a completion request with prompt_logprobs and optional temperature."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "prompt_logprobs": 1,  # return top-1 + selected token
        "temperature": 0,
    }
    if prompt_logprobs_temperature is not None:
        payload["prompt_logprobs_temperature"] = prompt_logprobs_temperature

    resp = requests.post(f"{base_url}/v1/completions", json=payload)
    resp.raise_for_status()
    return resp.json()


def avg_argmax_logprob(prompt_logprobs_list):
    """Compute average logprob of the argmax token across prompt positions."""
    logprobs = []
    for token_logprobs in prompt_logprobs_list:
        if token_logprobs is None:
            continue  # first token has no logprob
        # Find the max logprob across all returned tokens (= the argmax token)
        max_lp = max(info["logprob"] for info in token_logprobs.values())
        logprobs.append(max_lp)
    return sum(logprobs) / len(logprobs) if logprobs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "In the beginning, there was nothing but darkness and silence."
    )

    print("=" * 60)
    print("Test: Prompt Logprobs Temperature")
    print("=" * 60)
    print()
    print("Metric: average logprob of the ARGMAX token per position.")
    print("Low temp  → peaked distribution  → argmax logprob closer to 0")
    print("High temp → flat distribution    → argmax logprob more negative")

    # Test 1: No temperature (default / raw logits)
    print("\n[1] No prompt_logprobs_temperature (default)...")
    resp_default = get_prompt_logprobs(base_url, args.model, prompt)
    lp_default = resp_default["choices"][0]["prompt_logprobs"]
    avg_default = avg_argmax_logprob(lp_default)
    print(f"    Avg argmax logprob: {avg_default:.4f}")

    # Test 2: Low temperature → argmax logprob should be HIGHEST (closest to 0)
    print("\n[2] prompt_logprobs_temperature = 0.1 (low)...")
    resp_low = get_prompt_logprobs(base_url, args.model, prompt,
                                   prompt_logprobs_temperature=0.1)
    lp_low = resp_low["choices"][0]["prompt_logprobs"]
    avg_low = avg_argmax_logprob(lp_low)
    print(f"    Avg argmax logprob: {avg_low:.4f}")

    # Test 3: High temperature → argmax logprob should be LOWEST (most negative)
    print("\n[3] prompt_logprobs_temperature = 5.0 (high)...")
    resp_high = get_prompt_logprobs(base_url, args.model, prompt,
                                    prompt_logprobs_temperature=5.0)
    lp_high = resp_high["choices"][0]["prompt_logprobs"]
    avg_high = avg_argmax_logprob(lp_high)
    print(f"    Avg argmax logprob: {avg_high:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Default (no temp):  avg argmax logprob = {avg_default:.4f}")
    print(f"  Low temp (0.1):     avg argmax logprob = {avg_low:.4f}")
    print(f"  High temp (5.0):    avg argmax logprob = {avg_high:.4f}")
    print()

    # Low temp → more peaked → argmax gets higher logprob (closer to 0)
    # High temp → flatter → argmax gets lower logprob (more negative)
    if avg_low > avg_default > avg_high:
        print("✅ PASS: argmax_low > argmax_default > argmax_high")
    elif avg_low > avg_high:
        print("⚠️  PARTIAL: argmax_low > argmax_high (temp has effect)")
        print("   Default didn't fall in between (expected since default ≈ temp=1.0)")
    else:
        print("❌ FAIL: Expected argmax_low > argmax_high")

    if avg_low == avg_default == avg_high:
        print("❌ FAIL: All values identical — temperature not being applied")

    print("=" * 60)


if __name__ == "__main__":
    main()

"""
# start server:
export VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 

# install env: 
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
uv pip install pdbpp
# """