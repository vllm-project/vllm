#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Collect REAL-DIVERGENCE training data for the SSD OutcomePredictor.

Unlike the self-play script (collect_training_data.py), this script creates
genuine draft/target divergence by using two separate forward passes with
different temperature parameters:

  Draft distribution:  T=1.5  (higher temp → flatter, more random)
  Target distribution: T=0.7  (lower temp → peaked, more correct)

This matches realistic speculative decoding where:
  - Draft model is a smaller/faster model with higher entropy
  - Target model is more confident/correct

Expected acceptance rates (per position):
  pos0 ≈ 0.65-0.75   (usually good first token)
  pos1 ≈ 0.45-0.55
  pos2 ≈ 0.30-0.40
  pos3 ≈ 0.20-0.30

This is the proper training signal for the SSD OutcomePredictor.

Usage:
    python collect_divergence_data.py \
        --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output training_data_divergence.pt \
        --num-steps 2000
"""

from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn.functional as F

HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
    "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)

DEFAULT_OUTPUT = (
    "/home/buddywhitman/mpi_workspace/project3_ssd_vllm/training_data_divergence.pt"
)
DEFAULT_NUM_STEPS = 2000
K = 4
DRAFT_TEMP = 1.5   # Higher → more diverse draft, more rejections
TARGET_TEMP = 0.7  # Lower  → more peaked target, reflects confident target model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of artificial intelligence dates back to ",
    "In quantum computing, researchers have discovered that ",
    "Machine learning models are trained on massive datasets to ",
    "The capital of France is Paris, which is renowned for its ",
    "Python is a high-level programming language known for its ",
    "The theory of relativity, proposed by Einstein, states that ",
    "Natural language processing enables computers to understand ",
    "Modern neural networks are inspired by biological neurons, which ",
    "The immune system protects the body from pathogens by ",
    "Climate change is primarily caused by greenhouse gases, which ",
    "The discovery of penicillin revolutionized medicine by ",
    "Blockchain technology ensures data integrity through ",
    "The human genome contains approximately 3 billion base pairs that ",
    "Renewable energy sources like solar and wind are important because ",
    "Deep learning architectures like transformers have enabled ",
    "The laws of thermodynamics describe how energy behaves in ",
    "Cognitive science combines psychology, neuroscience, and AI to ",
    "The Standard Model of particle physics explains ",
    "Evolutionary biology shows that species adapt to their environment by ",
    "The Internet of Things connects physical devices to networks by ",
]


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use local cache if available
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        if os.path.exists(os.path.join(HF_CACHE, "model.safetensors")):
            print(f"Using cached model at {HF_CACHE}")
            model_path = HF_CACHE
        else:
            print(f"Falling back to HF hub: {HF_MODEL}")
            model_path = HF_MODEL

    print(f"Loading TinyLlama from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        output_hidden_states=True,
    ).to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {n_params:.0f}M params on {DEVICE}")
    return model, tokenizer


@torch.no_grad()
def draft_k_tokens(
    model,
    input_ids: torch.Tensor,
    K: int,
    temperature: float = DRAFT_TEMP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draft K tokens autoregressively at high temperature."""
    current_ids = input_ids.clone()
    draft_tokens, draft_logits_list = [], []
    hidden_state = None

    for step in range(K):
        outputs = model(current_ids.unsqueeze(0), output_hidden_states=True)
        logits = outputs.logits[0, -1, :]

        if step == 0:
            hidden_state = outputs.hidden_states[-1][0, -1, :].detach().clone()

        draft_logits_list.append(logits.detach().clone())

        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        draft_tokens.append(token)
        current_ids = torch.cat([current_ids, token.unsqueeze(0)], dim=0)

    return (
        torch.stack(draft_tokens),
        torch.stack(draft_logits_list),
        hidden_state,
    )


@torch.no_grad()
def target_verify(
    model,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_temp: float = TARGET_TEMP,
) -> torch.Tensor:
    """
    Verify draft tokens against target distribution.

    Target uses lower temperature (more confident), creating real divergence.
    Applies standard rejection sampling: accept_k = (u < p_target_k / p_draft_k)
    """
    K = draft_tokens.shape[0]
    verify_ids = torch.cat([input_ids, draft_tokens], dim=0)
    outputs = model(verify_ids.unsqueeze(0))
    start = len(input_ids) - 1
    target_logits = outputs.logits[0, start : start + K, :]

    # Target uses lower temperature → more peaked distribution
    target_probs = F.softmax(target_logits / target_temp, dim=-1)
    # Draft probs at draft temp
    draft_probs = F.softmax(draft_logits / DRAFT_TEMP, dim=-1)

    accept_mask = torch.zeros(K, dtype=torch.bool, device=DEVICE)
    for k in range(K):
        token = draft_tokens[k].item()
        p_t = target_probs[k, token].item()
        p_d = draft_probs[k, token].item()
        u = random.random()
        if u < min(1.0, p_t / (p_d + 1e-10)):
            accept_mask[k] = True
        else:
            # Cascade rejection
            break

    return accept_mask


def collect(model, tokenizer, num_steps: int) -> list[dict]:
    data = []
    prompt_idx = 0
    prompt = PROMPTS[prompt_idx % len(PROMPTS)]
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].to(DEVICE)

    print(f"\nCollecting {num_steps} divergence steps "
          f"(draft_T={DRAFT_TEMP}, target_T={TARGET_TEMP}, K={K})")
    print(f"Expected acceptance: pos0≈0.70, pos1≈0.50, pos2≈0.35, pos3≈0.22")

    for step in range(num_steps):
        if step % 200 == 0:
            acc_so_far = 0.0
            if data:
                all_m = torch.stack([d["accept_mask"] for d in data])
                acc_so_far = all_m.float().mean().item()
            print(f"  step {step:4d}/{num_steps} | "
                  f"ctx={len(input_ids):3d} | acc={acc_so_far:.3f}")

        if len(input_ids) > 300:
            prompt_idx += 1
            prompt = PROMPTS[prompt_idx % len(PROMPTS)]
            input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].to(DEVICE)

        try:
            draft_tokens, draft_logits, hidden = draft_k_tokens(
                model, input_ids, K)
            accept_mask = target_verify(
                model, input_ids, draft_tokens, draft_logits)

            data.append({
                "draft_logits": draft_logits.cpu().float(),
                "hidden_state": hidden.cpu().float(),
                "accept_mask": accept_mask.cpu(),
            })

            # Advance: accepted tokens + correction
            n = int(accept_mask.sum().item())
            new_tokens = draft_tokens[:n]
            if n < K:
                new_tokens = torch.cat([new_tokens, draft_tokens[n:n+1]])
            else:
                new_tokens = torch.cat([new_tokens, draft_tokens[-1:]])
            input_ids = torch.cat([input_ids, new_tokens])

        except Exception as e:
            print(f"  Warning step {step}: {e}")
            prompt_idx += 1
            prompt = PROMPTS[prompt_idx % len(PROMPTS)]
            input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].to(DEVICE)

    if data:
        all_m = torch.stack([d["accept_mask"] for d in data])
        print(f"\nResults: {len(data)} samples")
        print(f"  Per-position acceptance:")
        for k in range(K):
            rate = all_m[:, k].float().mean().item()
            print(f"    pos{k}: {rate:.3f}")
        print(f"  Mean acceptance: {all_m.float().mean().item():.3f}")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=HF_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = load_model(args.model_path)
    data = collect(model, tokenizer, args.num_steps)

    if not data:
        print("ERROR: No data collected"); return 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"\nSaved {len(data)} samples → {args.output}")
    sample = data[0]
    print(f"  draft_logits: {sample['draft_logits'].shape}")
    print(f"  hidden_state: {sample['hidden_state'].shape}")
    print(f"  accept_mask:  {sample['accept_mask'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
