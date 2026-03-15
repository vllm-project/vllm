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
uv pip install pdbpp pytest pytest_asyncio tblib

# pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/v1/sample/test_logprobs.py -v -s -k prompt_logprobs_temperature_increases_argmax_prob

python -m pytest tests/v1/entrypoints/openai/test_completion.py -v -s -k test_prompt_logprobs_temperature_increases_argmax_prob_completion


Minimal test: compare prompt logprob probabilities at different temperatures.

Usage:
  1. Start a vLLM server:
     python -m vllm.entrypoints.openai.api_server \
         --model <your-model> --port 8000

  2. Run this script:
     python test_prompt_logprobs_temp.py [--model <model-name>] [--port 8000]
"""
import argparse
import math
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


def extract_selected_probs(prompt_logprobs_list):
    """Extract the probability of the selected (actual next) token at each position."""
    probs = []
    for token_logprobs in prompt_logprobs_list:
        if token_logprobs is None:
            continue
        # First entry is the selected token
        for token_info in token_logprobs.values():
            probs.append(math.exp(token_info["logprob"]))
            break
    return probs


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

    print("=" * 70)
    print("Test: Prompt Logprobs Temperature — Token-Level Analysis")
    print("=" * 70)

    # Fetch probabilities at three temperature settings
    print("\nFetching prompt logprobs...")

    resp_low = get_prompt_logprobs(base_url, args.model, prompt,
                                   prompt_logprobs_temperature=0.1)
    probs_low = extract_selected_probs(
        resp_low["choices"][0]["prompt_logprobs"]
    )

    resp_med = get_prompt_logprobs(base_url, args.model, prompt)
    probs_med = extract_selected_probs(
        resp_med["choices"][0]["prompt_logprobs"]
    )

    resp_high = get_prompt_logprobs(base_url, args.model, prompt,
                                    prompt_logprobs_temperature=5.0)
    probs_high = extract_selected_probs(
        resp_high["choices"][0]["prompt_logprobs"]
    )

    n = min(len(probs_low), len(probs_med), len(probs_high))

    # Print per-token comparison
    print(f"\n{'Pos':>4}  {'Low(0.1)':>10}  {'Med(1.0)':>10}  {'High(5.0)':>10}  "
          f"{'Low>Med':>8}  {'Med>High':>9}")
    print("-" * 70)
    for i in range(n):
        pl, pm, ph = probs_low[i], probs_med[i], probs_high[i]
        low_gt_med = "✓" if pl > pm else ""
        med_gt_high = "✓" if pm > ph else ""
        print(f"{i:4d}  {pl:10.6f}  {pm:10.6f}  {ph:10.6f}  "
              f"{low_gt_med:>8}  {med_gt_high:>9}")

    # Token-level statistics: low temp vs medium temp
    low_gt_med_count = sum(1 for i in range(n) if probs_low[i] > probs_med[i])
    low_le_med_count = n - low_gt_med_count

    avg_low_temp_prob_when_gt = (
        sum(probs_low[i] for i in range(n) if probs_low[i] > probs_med[i])
        / low_gt_med_count if low_gt_med_count > 0 else 0
    )
    avg_med_temp_prob_when_lt = (
        sum(probs_med[i] for i in range(n) if probs_low[i] > probs_med[i])
        / low_gt_med_count if low_gt_med_count > 0 else 0
    )
    avg_low_temp_prob_when_le = (
        sum(probs_low[i] for i in range(n) if probs_low[i] <= probs_med[i])
        / low_le_med_count if low_le_med_count > 0 else 0
    )
    avg_med_temp_prob_when_ge = (
        sum(probs_med[i] for i in range(n) if probs_low[i] <= probs_med[i])
        / low_le_med_count if low_le_med_count > 0 else 0
    )

    print(f"\n{'=' * 70}")
    print("Token-Level Statistics: Low Temp (0.1) vs Medium Temp (1.0)")
    print(f"{'=' * 70}")
    print(f"  Tokens where low_temp_prob > med_temp_prob:  "
          f"{low_gt_med_count}/{n} ({100*low_gt_med_count/n:.1f}%)")
    print(f"    Avg low_temp_prob in these cases: {avg_low_temp_prob_when_gt:.6f}")
    print(f"    Avg med_temp_prob in these cases: {avg_med_temp_prob_when_lt:.6f}")
    print(f"  Tokens where low_temp_prob <= med_temp_prob: "
          f"{low_le_med_count}/{n} ({100*low_le_med_count/n:.1f}%)")
    print(f"    Avg low_temp_prob in these cases: {avg_low_temp_prob_when_le:.6f}")
    print(f"    Avg med_temp_prob in these cases: {avg_med_temp_prob_when_ge:.6f}")

    # Token-level statistics: medium temp vs high temp
    med_gt_high_count = sum(1 for i in range(n) if probs_med[i] > probs_high[i])
    med_le_high_count = n - med_gt_high_count

    avg_med_temp_prob_when_gt = (
        sum(probs_med[i] for i in range(n) if probs_med[i] > probs_high[i])
        / med_gt_high_count if med_gt_high_count > 0 else 0
    )
    avg_high_temp_prob_when_lt = (
        sum(probs_high[i] for i in range(n) if probs_med[i] > probs_high[i])
        / med_gt_high_count if med_gt_high_count > 0 else 0
    )
    avg_med_temp_prob_when_le = (
        sum(probs_med[i] for i in range(n) if probs_med[i] <= probs_high[i])
        / med_le_high_count if med_le_high_count > 0 else 0
    )
    avg_high_temp_prob_when_ge = (
        sum(probs_high[i] for i in range(n) if probs_med[i] <= probs_high[i])
        / med_le_high_count if med_le_high_count > 0 else 0
    )

    print(f"\n{'=' * 70}")
    print("Token-Level Statistics: Medium Temp (1.0) vs High Temp (5.0)")
    print(f"{'=' * 70}")
    print(f"  Tokens where med_temp_prob > high_temp_prob:  "
          f"{med_gt_high_count}/{n} ({100*med_gt_high_count/n:.1f}%)")
    print(f"    Avg med_temp_prob  in these cases: {avg_med_temp_prob_when_gt:.6f}")
    print(f"    Avg high_temp_prob in these cases: {avg_high_temp_prob_when_lt:.6f}")
    print(f"  Tokens where med_temp_prob <= high_temp_prob: "
          f"{med_le_high_count}/{n} ({100*med_le_high_count/n:.1f}%)")
    print(f"    Avg med_temp_prob  in these cases: {avg_med_temp_prob_when_le:.6f}")
    print(f"    Avg high_temp_prob in these cases: {avg_high_temp_prob_when_ge:.6f}")

    # Overall averages
    avg_p_low = sum(probs_low[:n]) / n
    avg_p_med = sum(probs_med[:n]) / n
    avg_p_high = sum(probs_high[:n]) / n

    print(f"\n{'=' * 70}")
    print("Overall Average Probabilities")
    print(f"{'=' * 70}")
    print(f"  Low temp (0.1):      {avg_p_low:.6f}")
    print(f"  Medium temp (1.0):   {avg_p_med:.6f}")
    print(f"  High temp (5.0):     {avg_p_high:.6f}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("Interpretation")
    print(f"{'=' * 70}")
    print("Low temp sharpens the distribution:")
    print("  - High-prob tokens (already likely) get HIGHER probability")
    print("  - Low-prob tokens (unlikely) get LOWER probability")
    print("High temp flattens the distribution:")
    print("  - All tokens move toward uniform (1/vocab_size)")
    print(f"\nKey check: Do all three settings produce DIFFERENT values?")
    all_different = (avg_p_low != avg_p_med and
                     avg_p_med != avg_p_high and
                     avg_p_low != avg_p_high)
    if all_different:
        print("  ✅ YES — temperature is being applied correctly")
    else:
        print("  ❌ NO — temperature may not be working")

    print("=" * 70)


if __name__ == "__main__":
    main()
