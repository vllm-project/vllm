"""Calibrate skip-softmax threshold scale factors with NVIDIA Model-Optimizer.

Runs RULER-based calibration of the ``flash_skip_softmax`` method from
``nvidia-modelopt`` against a HuggingFace causal-LM, then writes a JSON
file containing the ``(a, b)`` exponential-model parameters per phase
plus the concrete scale factors to pass to vLLM:

    scale_factor(target) = a * exp(b * target)

which is exactly the value of
``--attention-config.skip_softmax_threshold_scale_factor_{prefill,decode}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--out",
        default="calibration/skip_softmax.json",
        help="Path to write calibration JSON",
    )
    p.add_argument(
        "--target-sparsity-prefill",
        type=float,
        default=0.7,
        help="Target prefill sparsity (0.0 skips prefill calibration)",
    )
    p.add_argument(
        "--target-sparsity-decode",
        type=float,
        default=0.7,
        help="Target decode sparsity (0.0 skips decode calibration)",
    )
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--max-seqlen", type=int, default=16384)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--num-decode-tokens", type=int, default=10)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache generated RULER samples (optional)",
    )
    p.add_argument(
        "--also-evaluate",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Additional target sparsities to report scale factors for "
            "(does not re-run calibration). Example: --also-evaluate 0.3 0.5 0.7"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    t0 = time.time()
    print(f"[calibrate] Loading {args.model} with attn_implementation='eager' …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[calibrate] Loaded in {time.time() - t0:.1f}s")

    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.sparsity.attention_sparsity.calibration import calibrate as _cal
    from modelopt.torch.sparsity.attention_sparsity.utils import (
        get_named_sparse_attention_modules,
    )

    # Monkey-patch the decode-phase fast-prefill attn impl: transformers 5.6
    # ships a buggy flash_attention path for Qwen3-MoE (s_aux can be None).
    # SDPA avoids F.softmax too, so it still bypasses measurement.
    _orig = _cal.create_decode_calibration_forward_loop

    def _patched_decode_loop(*a, **kw):
        inner = _orig(*a, **kw)

        def wrapped(model):
            orig_cfg_value = getattr(model.config, "_attn_implementation", "eager")
            # Replace the function's closure-written "flash_attention_2" with
            # "sdpa" by substituting the model's config right before inner runs.
            try:
                return inner(model)
            finally:
                model.config._attn_implementation = orig_cfg_value

        return wrapped

    # Simpler: directly patch the hardcoded backend string in the factory.
    import modelopt.torch.sparsity.attention_sparsity.calibration.calibrate as _calmod

    _src_fn = _calmod.create_decode_calibration_forward_loop

    def _safer_decode_factory(calibration_data, tokenizer_name_or_path, num_decode_tokens=10):
        # Mirror the original implementation but use "sdpa" for fast prefill.
        from modelopt.torch.utils import get_module_device

        tok = _calmod._load_tokenizer(tokenizer_name_or_path)

        def forward_loop(model):
            device = get_module_device(model)
            for sample in calibration_data:
                inputs = tok(
                    sample["input"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=sample["length"],
                )
                input_ids = inputs["input_ids"].to(device)
                original = getattr(model.config, "_attn_implementation", "eager")
                with torch.no_grad():
                    try:
                        model.config._attn_implementation = "sdpa"
                        outputs = model(input_ids, use_cache=True)
                        past_kv = outputs.past_key_values
                        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                        del outputs

                        model.config._attn_implementation = "eager"
                        for _ in range(num_decode_tokens):
                            outputs = model(
                                next_token,
                                past_key_values=past_kv,
                                use_cache=True,
                            )
                            past_kv = outputs.past_key_values
                            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                            del outputs
                    finally:
                        model.config._attn_implementation = original
                del past_kv
                torch.cuda.empty_cache()

        return forward_loop

    _calmod.create_decode_calibration_forward_loop = _safer_decode_factory

    sparse_cfg: dict = {
        "*": {
            "method": "flash_skip_softmax",
            "backend": "pytorch",
            "enable": True,
            "br": 128,
            "bc": 128,
            "is_causal": True,
        },
        "calibration": {
            "target_sparse_ratio": {
                "prefill": args.target_sparsity_prefill,
                "decode": args.target_sparsity_decode,
            },
            "samples": args.samples,
            "max_seqlen": args.max_seqlen,
            "chunk_size": args.chunk_size,
            "num_decode_tokens": args.num_decode_tokens,
        },
    }
    if args.cache_dir:
        sparse_cfg["calibration"]["cache_dir"] = args.cache_dir

    config = {"sparse_cfg": sparse_cfg}

    print("[calibrate] Starting calibration …")
    t1 = time.time()
    mtsa.sparsify(model, config)
    print(f"[calibrate] Calibration finished in {time.time() - t1:.1f}s")

    named = get_named_sparse_attention_modules(model)
    if not named:
        print("[calibrate] ERROR: no sparse attention modules registered", file=sys.stderr)
        return 2
    _, any_mod = named[0]
    method = any_mod._sparse_method_instance
    params = getattr(method, "calibration_params", None)
    targets = getattr(method, "target_sparse_ratio", None)

    if not params:
        print("[calibrate] ERROR: calibration produced no parameters", file=sys.stderr)
        return 3

    def scale(phase: str, target: float) -> float:
        p_ = params[phase]
        return float(p_["a"] * math.exp(p_["b"] * target))

    result: dict = {
        "model": args.model,
        "calibration_params": {
            phase: {k: float(v) for k, v in params[phase].items()}
            for phase in params
        },
        "target_sparse_ratio": {k: float(v) for k, v in targets.items()},
        "formula": "scale_factor = a * exp(b * target_sparsity)",
        "vllm_flags": {},
        "additional_operating_points": {},
    }

    if "prefill" in params and args.target_sparsity_prefill > 0:
        result["vllm_flags"]["skip_softmax_threshold_scale_factor_prefill"] = scale(
            "prefill", args.target_sparsity_prefill
        )
    if "decode" in params and args.target_sparsity_decode > 0:
        result["vllm_flags"]["skip_softmax_threshold_scale_factor_decode"] = scale(
            "decode", args.target_sparsity_decode
        )

    for extra in args.also_evaluate or []:
        result["additional_operating_points"][f"target={extra}"] = {
            phase: scale(phase, extra) for phase in params
        }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"[calibrate] Wrote {out_path}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
