#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Equivalence test: offline vs. online FP8 quantization (output logits).

For a given quant scheme we ship two checkpoints of the same model:

* an **offline** pre-quantized checkpoint (e.g. ``c5-3a30t-sft_mxfp8``), and
* an **online** checkpoint (e.g. ``c5-3a30t-sft_mxfp8_online``) whose
  ``config.json`` triggers quantize-at-load (``Mxfp8OnlineLinearMethod`` /
  ``Fp8PerBlockOnlineLinearMethod``).

If the offline and online paths implement the same quantization math, the two
checkpoints should produce essentially the same next-token distribution for the
same prompts. This test loads each checkpoint and, for every fixed prompt,
captures the **full-vocabulary logits of the next token** via a forward hook on
``logits_processor`` (the same hook pattern used by ``test_c5_fp32_logits.py``),
then compares the resulting probability distributions with proper distance
metrics:

* **L1 distance** ``||p_off - p_on||_1`` per prompt (bounded in ``[0, 2]``),
* **KL divergence** ``D_KL(p_off || p_on)`` per prompt,
* **top-1 agreement** (argmax match).

On top of the first-token distribution metrics, the test also asserts
**rollout identity**: under greedy decoding (over the short ``C5_QUANT_MAX_TOKENS``
horizon) the offline and online checkpoints must emit the exact same generated
token sequence for every prompt.

We capture exactly one row per prompt (the next-token / sampling position) so
the comparison is robust: the row count is ``num_prompts`` regardless of prompt
length, chunked prefill, or per-checkpoint auto-config -- so the two
checkpoints' captured matrices always align 1:1.

Driven entirely by environment variables so it is a no-op (skips) when the
checkpoints are not provisioned:

* ``C5_OFFLINE_MODEL_DIR`` / ``C5_ONLINE_MODEL_DIR`` (required) -- local dirs.
* ``C5_QUANT_SCHEME`` (optional) -- ``mxfp8`` / ``block_fp8`` / ``fp8``; used to
  skip on GPUs that do not support the scheme. ``mxfp8`` additionally only runs
  on Blackwell GPUs (skipped on older architectures).
* ``C5_QUANT_TENSOR_PARALLEL_SIZE`` (default 1).
* ``C5_QUANT_MAX_L1`` / ``C5_QUANT_MEAN_L1`` / ``C5_QUANT_MIN_TOP1_AGREE``
  (tolerance overrides).
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.platforms import current_platform

from .test_utils_c5 import (
    C5_SANITY_PROMPTS,
    shutdown_llm,
    validate_model_path,
)


def _is_blackwell() -> bool:
    """True on NVIDIA Blackwell GPUs (compute capability major >= 10, i.e.
    sm_100 datacenter parts like B200/GB200 and sm_120 consumer parts)."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major >= 10


def _tensor_parallel_size() -> int:
    raw_value = os.environ.get("C5_QUANT_TENSOR_PARALLEL_SIZE")
    if raw_value is None:
        return 1
    return int(raw_value)


def _max_l1() -> float:
    raw_value = os.environ.get("C5_QUANT_MAX_L1")
    return float(raw_value) if raw_value is not None else 0


def _mean_l1() -> float:
    raw_value = os.environ.get("C5_QUANT_MEAN_L1")
    return float(raw_value) if raw_value is not None else 0


def _min_top1_agreement() -> float:
    raw_value = os.environ.get("C5_QUANT_MIN_TOP1_AGREE")
    return float(raw_value) if raw_value is not None else 1


def _max_tokens() -> int:
    # The first decode step drives the distribution metrics; the remaining
    # tokens form the greedy rollout compared for exact equality.
    raw_value = os.environ.get("C5_QUANT_MAX_TOKENS")
    return int(raw_value) if raw_value is not None else 16


def _get_logits_processor(model):
    logits_processor = getattr(model, "logits_processor", None)
    if logits_processor is not None:
        return logits_processor

    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return getattr(language_model, "logits_processor", None)

    return None


def _install_logits_capture_hook(model) -> bool:
    """Install (once) a forward hook on ``logits_processor`` that records the
    full-vocab logits tensor for every position it is invoked on."""
    logits_processor = _get_logits_processor(model)
    if logits_processor is None:
        return False

    # Reset the buffer so repeated captures within a process start clean.
    logits_processor._captured_logits = []
    if getattr(logits_processor, "_capture_hook_installed", False):
        return True

    def _capture(module, _inputs, output) -> None:
        if output is not None:
            module._captured_logits.append(output.detach().float().cpu())

    logits_processor.register_forward_hook(_capture)
    logits_processor._capture_hook_installed = True
    return True


def _reset_capture_buffer(model) -> bool:
    logits_processor = _get_logits_processor(model)
    if logits_processor is not None:
        logits_processor._captured_logits = []
    return True


def _collect_captured_logits(model):
    logits_processor = _get_logits_processor(model)
    if logits_processor is None:
        return None
    chunks = getattr(logits_processor, "_captured_logits", None)
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def _first_non_none(values):
    for value in values:
        if value is not None:
            return value
    return None


def _capture_logits(
    model_path: str,
    tensor_parallel_size: int,
    prompt_token_ids: list[list[int]] | None = None,
) -> tuple[torch.Tensor, list[dict], list[list[int]]]:
    """Load ``model_path`` and return the next-token (full-vocab) logits after
    each prompt as a ``[num_prompts, vocab]`` matrix, a per-prompt diagnostic
    record, and the input token-ids used.

    Inputs are fed as **token ids** (``TokensPrompt``), not raw strings: the two
    checkpoints can ship different tokenizer ``post_processor`` templates (e.g.
    one appends ``<|END_OF_TURN_TOKEN|><EOS_TOKEN>``), so tokenizing each side
    with its own tokenizer would feed the two models different sequences. To
    isolate the weight/quant comparison, the caller tokenizes the prompts once
    (with the first/offline model's tokenizer, ``prompt_token_ids=None``) and
    passes the SAME ids to the other model.

    We deliberately do **not** request ``prompt_logprobs``: the number of
    prompt-position logits the hook sees is not stable across checkpoints, which
    would break position-wise alignment. Instead we run one prompt at a time and
    take the **first** captured row -- the logits used to sample the first new
    token -- exactly one full-vocab distribution per prompt, independent of
    prompt length / prefill internals. ``max_tokens`` is > 1
    (``C5_QUANT_MAX_TOKENS``) so the recorded ``gen_token_ids`` form a multi-step
    greedy rollout; the caller compares those sequences for exact equality.
    """
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_max_tokens())
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    rows: list[torch.Tensor] = []
    records: list[dict] = []
    try:
        if prompt_token_ids is None:
            tokenizer = llm.get_tokenizer()
            prompt_token_ids = [
                list(tokenizer.encode(prompt)) for prompt in C5_SANITY_PROMPTS
            ]

        installed = llm.apply_model(_install_logits_capture_hook)
        if not any(installed):
            raise RuntimeError(
                f"Could not locate logits_processor to hook for {model_path}."
            )
        for token_ids in prompt_token_ids:
            llm.apply_model(_reset_capture_buffer)
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=token_ids)],
                sampling_params=sampling_params,
            )
            captured = _first_non_none(llm.apply_model(_collect_captured_logits))
            if captured is None or captured.shape[0] == 0:
                raise RuntimeError(
                    f"No logits captured for a prompt under {model_path}."
                )
            # The first captured row is the next-token (post-prompt) position;
            # later rows are subsequent decode steps and are not used here.
            rows.append(captured[0])

            completion = outputs[0].outputs[0]
            records.append(
                {
                    "prompt_token_ids": list(token_ids),
                    "gen_token_ids": [int(t) for t in completion.token_ids],
                    "gen_text": completion.text,
                }
            )
    finally:
        shutdown_llm(llm)

    return torch.stack(rows, dim=0), records, prompt_token_ids


def _distribution_metrics(
    offline_logits: torch.Tensor, online_logits: torch.Tensor
) -> dict[str, float]:
    # Compute in float64 for numerically stable softmax / KL.
    offline_logits = offline_logits.double()
    online_logits = online_logits.double()

    log_p_off = torch.log_softmax(offline_logits, dim=-1)
    log_p_on = torch.log_softmax(online_logits, dim=-1)
    p_off = log_p_off.exp()
    p_on = log_p_on.exp()

    l1 = (p_off - p_on).abs().sum(dim=-1)  # [N], in [0, 2]
    kl = (p_off * (log_p_off - log_p_on)).sum(dim=-1).clamp_min(0.0)  # [N]
    top1 = (offline_logits.argmax(dim=-1) == online_logits.argmax(dim=-1)).double()

    return {
        "num_positions": float(offline_logits.shape[0]),
        "vocab_size": float(offline_logits.shape[1]),
        "max_l1": float(l1.max()),
        "mean_l1": float(l1.mean()),
        "max_kl": float(kl.max()),
        "mean_kl": float(kl.mean()),
        "top1_agreement": float(top1.mean()),
    }


def test_c5_online_vs_offline_quant_logits() -> None:
    offline_dir = os.environ.get("C5_OFFLINE_MODEL_DIR")
    online_dir = os.environ.get("C5_ONLINE_MODEL_DIR")
    if not offline_dir or not online_dir:
        pytest.skip(
            "C5_OFFLINE_MODEL_DIR and C5_ONLINE_MODEL_DIR must be set to run the "
            "offline-vs-online quant equivalence test."
        )
    assert offline_dir is not None
    assert online_dir is not None

    # Skip on hardware that does not support the scheme under test (e.g. mxfp8
    # needs sm_100+). block_fp8 / fp8 both map to the "fp8" capability check.
    scheme = os.environ.get("C5_QUANT_SCHEME")
    if scheme:
        # mxfp8 equivalence is only validated on Blackwell; skip on older archs.
        if scheme == "mxfp8" and not _is_blackwell():
            pytest.skip("Quant scheme 'mxfp8' is only tested on Blackwell GPUs.")
        support_key = "fp8" if scheme in ("fp8", "block_fp8") else scheme
        if not is_quant_method_supported(support_key):
            pytest.skip(f"Quant scheme {scheme!r} is not supported on this GPU.")

    offline_dir = validate_model_path(offline_dir)
    online_dir = validate_model_path(online_dir)
    tensor_parallel_size = _tensor_parallel_size()

    # apply_model ships a Python callable over RPC to the workers.
    insecure_key = "VLLM_ALLOW_INSECURE_SERIALIZATION"
    original_insecure = os.environ.get(insecure_key)
    os.environ[insecure_key] = "1"
    try:
        # Tokenize once with the offline model's tokenizer, then feed the SAME
        # token ids to the online model so both see identical inputs (the two
        # checkpoints' tokenizers may differ -- see _capture_logits docstring).
        offline_logits, offline_recs, shared_token_ids = _capture_logits(
            offline_dir, tensor_parallel_size
        )
        online_logits, online_recs, _ = _capture_logits(
            online_dir, tensor_parallel_size, prompt_token_ids=shared_token_ids
        )
    finally:
        if original_insecure is None:
            os.environ.pop(insecure_key, None)
        else:
            os.environ[insecure_key] = original_insecure

    # Per-prompt diagnostics. Both sides receive identical input token ids, so a
    # remaining divergence points at the weights/quant path: coherent-but-
    # different (or garbage) generations mean the weights are not equivalent.
    print("\nPer-prompt diagnostics (offline vs online; identical input token ids):")
    for idx, prompt in enumerate(C5_SANITY_PROMPTS):
        off, on = offline_recs[idx], online_recs[idx]
        print(
            f"\n  [{idx}] prompt: {prompt!r}\n"
            f"      input_token_ids (len={len(off['prompt_token_ids'])}): "
            f"{off['prompt_token_ids']}\n"
            f"      offline gen_token_ids={off['gen_token_ids']}\n"
            f"      offline gen_text={off['gen_text']!r}\n"
            f"      online  gen_token_ids={on['gen_token_ids']}\n"
            f"      online  gen_text={on['gen_text']!r}"
        )

    # Rollout-identity check: under greedy decoding the offline and online
    # checkpoints should emit the SAME generated token sequence for every prompt
    # (not just match on the first token). Mismatches indicate the two quant
    # paths diverge once small first-token differences compound over decode.
    rollout_mismatches = []
    for idx, prompt in enumerate(C5_SANITY_PROMPTS):
        off_ids = offline_recs[idx]["gen_token_ids"]
        on_ids = online_recs[idx]["gen_token_ids"]
        if off_ids != on_ids:
            divergence = next(
                (pos for pos, (a, b) in enumerate(zip(off_ids, on_ids)) if a != b),
                min(len(off_ids), len(on_ids)),
            )
            rollout_mismatches.append((idx, prompt, divergence))
    print(
        "\nRollout identity (offline vs online greedy generations): "
        f"{len(C5_SANITY_PROMPTS) - len(rollout_mismatches)}/"
        f"{len(C5_SANITY_PROMPTS)} prompts identical."
    )
    for idx, prompt, divergence in rollout_mismatches:
        print(f"  [{idx}] diverges at gen position {divergence}: {prompt!r}")

    assert offline_logits.shape == online_logits.shape, (
        "Captured logits shapes differ between offline and online runs "
        f"({tuple(offline_logits.shape)} vs {tuple(online_logits.shape)}); "
        "cannot compare position-wise."
    )

    metrics = _distribution_metrics(offline_logits, online_logits)

    max_l1_tol = _max_l1()
    mean_l1_tol = _mean_l1()
    min_top1 = _min_top1_agreement()

    print(
        "\nOffline vs online quant logits equivalence "
        f"(scheme={scheme or 'unknown'}, positions={int(metrics['num_positions'])}, "
        f"vocab={int(metrics['vocab_size'])}):\n"
        f"  max L1   : {metrics['max_l1']:.6f} (tol {max_l1_tol})\n"
        f"  mean L1  : {metrics['mean_l1']:.6f} (tol {mean_l1_tol})\n"
        f"  max KL   : {metrics['max_kl']:.6f}\n"
        f"  mean KL  : {metrics['mean_kl']:.6f}\n"
        f"  top1 agree: {metrics['top1_agreement']:.6f} (min {min_top1})"
    )

    failures = []
    if metrics["max_l1"] > max_l1_tol:
        failures.append(f"max L1 {metrics['max_l1']:.6f} > {max_l1_tol}")
    if metrics["mean_l1"] > mean_l1_tol:
        failures.append(f"mean L1 {metrics['mean_l1']:.6f} > {mean_l1_tol}")
    if metrics["top1_agreement"] < min_top1:
        failures.append(f"top1 agreement {metrics['top1_agreement']:.6f} < {min_top1}")
    if rollout_mismatches:
        prompt_indices = ", ".join(str(idx) for idx, _, _ in rollout_mismatches)
        failures.append(
            f"rollout differs for {len(rollout_mismatches)}/"
            f"{len(C5_SANITY_PROMPTS)} prompts (indices: {prompt_indices})"
        )

    assert not failures, (
        "Offline and online quantized checkpoints diverge: " + "; ".join(failures)
    )
