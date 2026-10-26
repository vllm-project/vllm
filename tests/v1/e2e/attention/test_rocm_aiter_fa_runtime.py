# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA through the engine: scheduler, model runners, graphs.

Each run builds a few-layer model from a flagship architecture with dummy
weights (attention dims unchanged, everything else shrunk) and compares its
logprobs with the same model on TRITON_ATTN under the same engine settings.
Hybrid models keep exactly their smallest repeating layer block:

- Qwen3: full-attention GQA (64/4/128).
- Muse Spark Open: sliding, full, sliding, sliding (128/16/64, window 2048),
  with the fused qk-norm+RoPE+KV insert that bypasses the backend's KV write.
- Qwen3.8-27B: 3 Gated DeltaNet layers + 1 full-attention layer (24/4/256).
- GPT-OSS-20B: sliding (128) + full layer, both with attention sinks.
- Gemma 4: refused at startup (its global layers use head size 512).

The batch mixes long prompts (chunked prefill -> extends), a shared prefix
(prefix-cache hits -> extends) and decode steps, so every phase runs under:

- CUDA graph mode: NONE, PIECEWISE, FULL_DECODE_ONLY, FULL_AND_PIECEWISE
- async scheduling on/off
- model runner v1 / v2
- KV cache dtype auto / fp8
- n-gram speculative decoding (multi-token decode rows)

Dummy weights make greedy tokens fragile, so the comparison is
``check_logprobs_close``: tokens may diverge only where the other run ranks
the token in its top-k.
"""

from dataclasses import dataclass, field, replace
from functools import partial

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCM_AITER_FA is ROCm only"
)


@dataclass(frozen=True)
class DummyModel:
    name: str
    hf_overrides: dict
    max_model_len: int = 2048
    # Prompt words; above the window for sliding-window models.
    long_words: int = 200
    # Extra LM-head gain that makes the reference logits clearly peaked.
    logit_gain: float = 8.0
    # Mean prompt-logprob difference allowed vs TRITON_ATTN, per KV dtype.
    # Calibrated on MI355 against a second correct backend (ROCM_ATTN or
    # ROCM_AITER_UNIFIED_ATTN): about a third of each bound.
    tolerance: dict = field(default_factory=lambda: {"auto": 0.15, "fp8": 0.6})
    # Greedy samples are compared with check_logprobs_close unless the model
    # is too noisy for it even between two correct backends.
    check_samples: bool = True
    engine_kwargs: dict = field(default_factory=dict)


# Qwen3-235B-A22B attention (64 q heads, 4 KV heads, head 128); MoE shrunk.
QWEN3 = DummyModel(
    "Qwen/Qwen3-235B-A22B",
    {
        "num_hidden_layers": 2,
        "hidden_size": 1024,
        "intermediate_size": 1024,
        "moe_intermediate_size": 256,
        # Every expert active: no routing ties for a backend difference to flip.
        "num_experts": 2,
        "num_experts_per_tok": 2,
    },
)
# Muse Spark Open attention (128 q heads, 16 KV heads, head 64); its repeating
# block is sliding (window 2048), full, sliding, sliding. MoE and vision tower
# shrunk. Gated on Hugging Face: needs HF_TOKEN.
MUSE_SPARK = DummyModel(
    "meta-models/Muse-Spark-Open-816B-A42B",
    {
        "text_config": {
            "num_hidden_layers": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
            "moe_hidden_size": 512,
            "intermediate_size": 1024,
        },
        "vision_config": {"num_hidden_layers": 2},
    },
    max_model_len=8192,
    long_words=1400,
    # Logits are scaled by output_multiplier (0.177) and soft-capped at 20; a
    # larger gain saturates the cap and ties the top tokens.
    logit_gain=20.0,
    # ROCM_ATTN differs from TRITON_ATTN by 0.10 (bf16) / 0.97 (fp8); a 1024
    # instead of 2048 window gives 1.58.
    tolerance={"auto": 0.4, "fp8": 1.5},
    engine_kwargs={"limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0}},
)
# Hybrid models use their smallest repeating layer block, so every layer kind
# and its neighbors appear once.
# Qwen3.8-27B: 3 Gated DeltaNet layers + 1 full-attention layer (24/4/256).
QWEN3_8 = DummyModel(
    "Qwen/Qwen3.8-27B",
    {
        "text_config": {"num_hidden_layers": 4, "intermediate_size": 1024},
        "vision_config": {"depth": 1},
    },
    engine_kwargs={"limit_mm_per_prompt": {"image": 0, "video": 0}},
)
# GPT-OSS-20B: sliding (128) + full layer, both with attention sinks (64/8/64).
GPT_OSS = DummyModel(
    "openai/gpt-oss-20b",
    {
        "num_hidden_layers": 2,
        "num_local_experts": 2,
        "num_experts_per_tok": 2,
        "experts_per_token": 2,
    },
    # MXFP4 experts with dummy bytes are noisier: ROCM_AITER_UNIFIED_ATTN is
    # 0.36 from TRITON_ATTN. With fp8 KV, prefill here attends over the exact
    # new K/V while TRITON_ATTN reads them back quantized (0.78).
    tolerance={"auto": 0.7, "fp8": 1.2},
    # Greedy decode diverges beyond top-5 even between ROCM_AITER_UNIFIED_ATTN
    # and TRITON_ATTN here; decode is covered by the other models.
    check_samples=False,
)
# Gemma 4 26B-A4B: 5 sliding (1024, 16/8/256) + 1 global (16/2/512) layer.
# ROCM_AITER_FA has no head size 512, so it cannot serve the global layers;
# its sliding shapes are covered by the unit suite.
GEMMA4 = DummyModel(
    "google/gemma-4-26B-A4B-it",
    {
        "text_config": {
            "num_hidden_layers": 6,
            "num_experts": 2,
            "top_k_experts": 2,
        },
        "vision_config": {"num_hidden_layers": 1},
    },
    max_model_len=4096,
    long_words=700,
    engine_kwargs={"limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0}},
)
MODELS = [
    pytest.param(QWEN3, id="qwen3"),
    pytest.param(MUSE_SPARK, id="muse_spark"),
    pytest.param(QWEN3_8, id="qwen3_8"),
    pytest.param(GPT_OSS, id="gpt_oss"),
]

MAX_TOKENS = 24
NUM_LOGPROBS = 5
SHARED_PREFIX = "You are a careful assistant. Answer briefly and cite sources. " * 8


def _prompts(model: DummyModel) -> list[str]:
    # Longer than max_num_batched_tokens, so it prefills in chunks (extends).
    long = " ".join(f"item{i}" for i in range(model.long_words))
    return [
        "Hello",
        SHARED_PREFIX + "What is the capital of France?",
        SHARED_PREFIX + "What is the capital of Japan?",
        long,
        "one two three one two three one two three one two three",
        "The quick brown fox jumps over the lazy dog. " * 20,
    ]


# Dummy weights are uniform in +-1e-3, which leaves logits almost flat (every
# logprob near -log(vocab)), so a broken attention would still "match".
DUMMY_HALF_RANGE = 1e-3


def _scale_dummy_weights(model: torch.nn.Module, logit_gain: float) -> None:
    """Give norms unit gain, rescale every matrix to std 1/sqrt(fan_in) and the
    LM head further by ``logit_gain``, in place and identically on every
    backend, so the logits are peaked and greedy tokens depend on what
    attention returns."""
    uniform_std = DUMMY_HALF_RANGE / 3**0.5
    for name, param in model.named_parameters():
        if param.dim() == 1 and "norm" in name:
            param.data.fill_(1.0)
        elif param.dim() >= 2 and param.dtype in (torch.bfloat16, torch.float16):
            gain = logit_gain if "lm_head" in name else 1.0
            param.data.mul_(gain * param.shape[-1] ** -0.5 / uniform_std)


def _bypassed_kv_writes(model: torch.nn.Module) -> int:
    """Attention layers whose KV write is done by a fused model op."""
    return sum(
        bool(getattr(m, "kv_cache_update_bypass", False)) for m in model.modules()
    )


def _run(
    vllm_runner,
    monkeypatch,
    *,
    backend,
    model=QWEN3,
    kv_cache_dtype="auto",
    cudagraph_mode="NONE",
    async_scheduling=False,
    model_runner_v2=False,
    speculative=False,
):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if model_runner_v2 else "0")
    kwargs = dict(model.engine_kwargs)
    if speculative:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
        }
    eager = cudagraph_mode == "NONE"
    with vllm_runner(
        model.name,
        load_format="dummy",
        hf_overrides=model.hf_overrides,
        dtype="bfloat16",
        max_model_len=model.max_model_len,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_dtype=kv_cache_dtype,
        attention_config={"backend": backend},
        enforce_eager=eager,
        compilation_config=None
        if eager
        else {
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        },
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.3,
        **kwargs,
    ) as llm:
        llm.apply_model(partial(_scale_dummy_weights, logit_gain=model.logit_gain))
        outputs = llm.generate_greedy_logprobs(
            _prompts(model), MAX_TOKENS, NUM_LOGPROBS, num_prompt_logprobs=0
        )
        return outputs, llm.apply_model(_bypassed_kv_writes)[0]


_REFERENCE: dict[tuple, list] = {}


def _mean_prompt_logprob_diff(ref: list, out: list) -> float:
    diffs = []
    for (*_, ref_prompt), (*_, out_prompt) in zip(ref, out):
        for ref_lp, out_lp in zip(ref_prompt[1:], out_prompt[1:]):
            token, logprob = next(iter(ref_lp.items()))
            diffs.append(abs(logprob.logprob - out_lp[token].logprob))
    return sum(diffs) / len(diffs)


def _reference(vllm_runner, monkeypatch, model, kv_cache_dtype, **kwargs):
    """TRITON_ATTN under the same engine settings (graph mode, runner,
    async, spec decode), so only the attention backend differs; cached."""
    key = (model.name, kv_cache_dtype, tuple(sorted(kwargs.items())))
    if key not in _REFERENCE:
        _REFERENCE[key], _ = _run(
            vllm_runner,
            monkeypatch,
            backend="TRITON_ATTN",
            model=model,
            kv_cache_dtype=kv_cache_dtype,
            **kwargs,
        )
    return _REFERENCE[key]


def _compare(vllm_runner, monkeypatch, model=QWEN3, kv_cache_dtype="auto", **kwargs):
    ref = _reference(vllm_runner, monkeypatch, model, kv_cache_dtype, **kwargs)
    out, bypassed = _run(
        vllm_runner,
        monkeypatch,
        backend="ROCM_AITER_FA",
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    )
    # The comparison only has power if the reference is far from uniform
    # (a uniform distribution gives about -12 for these vocabularies).
    top = [max(v.logprob for v in lp.values()) for _, _, lps, _ in ref for lp in lps]
    assert sum(top) / len(top) > -6.0, "reference logits are nearly flat"
    # Teacher-forced prompt logprobs cover every prefill/extend position,
    # including those past the sliding window. Calibrated on MI355: another
    # correct backend differs by <= 0.06 (bf16 KV) / 0.35 (fp8 KV) on average;
    # Muse Spark with a 1024 instead of 2048 window differs by 1.1.
    diff = _mean_prompt_logprob_diff(ref, out)
    tol = model.tolerance[kv_cache_dtype]
    assert diff <= tol, f"mean prompt logprob diff {diff:.3f} > {tol}"
    if not model.check_samples:
        return bypassed
    check_logprobs_close(
        outputs_0_lst=ref,
        outputs_1_lst=out,
        name_0="TRITON_ATTN",
        name_1="ROCM_AITER_FA",
    )
    return bypassed


CORE = [
    # (cudagraph_mode, async_scheduling, model_runner_v2)
    ("NONE", False, False),
    ("PIECEWISE", False, False),
    ("FULL_DECODE_ONLY", True, False),
    ("FULL_AND_PIECEWISE", True, False),
    ("PIECEWISE", False, True),
    ("FULL_DECODE_ONLY", True, True),
]
FULL = [
    (mode, async_, v2)
    for mode in ("NONE", "PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
    for async_ in (False, True)
    for v2 in (False, True)
    if (mode, async_, v2) not in CORE
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cudagraph_mode, async_scheduling, model_runner_v2", CORE)
def test_engine_modes(
    vllm_runner,
    monkeypatch,
    model,
    cudagraph_mode,
    async_scheduling,
    model_runner_v2,
):
    _compare(
        vllm_runner,
        monkeypatch,
        model=model,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cudagraph_mode", ["NONE", "FULL_AND_PIECEWISE"])
def test_fp8_kv_cache(vllm_runner, monkeypatch, model, cudagraph_mode):
    _compare(
        vllm_runner,
        monkeypatch,
        model=model,
        kv_cache_dtype="fp8",
        cudagraph_mode=cudagraph_mode,
    )


def test_muse_spark_takes_the_fused_kv_insert(vllm_runner, monkeypatch):
    """With ROCM_AITER_FA every Muse Spark layer writes its KV cache through
    the fused qk-norm+RoPE+KV insert; the backend then reads new tokens back
    from the cache (the KV-sharing workspace path)."""
    bypassed = _compare(vllm_runner, monkeypatch, model=MUSE_SPARK)
    assert bypassed == MUSE_SPARK.hf_overrides["text_config"]["num_hidden_layers"]


@pytest.mark.parametrize(
    "cudagraph_mode, model_runner_v2",
    [("NONE", False), ("FULL_AND_PIECEWISE", False), ("FULL_AND_PIECEWISE", True)],
)
def test_ngram_spec_decode(vllm_runner, monkeypatch, cudagraph_mode, model_runner_v2):
    # Runner v1 has no async scheduling with n-gram drafts.
    _compare(
        vllm_runner,
        monkeypatch,
        speculative=True,
        cudagraph_mode=cudagraph_mode,
        model_runner_v2=model_runner_v2,
        async_scheduling=model_runner_v2,
    )


def test_gemma4_global_layers_are_refused(vllm_runner, monkeypatch, capfd):
    """Forcing ROCM_AITER_FA on Gemma 4 must fail at startup with the reason
    (its global layers use head size 512), not at the first forward."""
    with pytest.raises(Exception, match="initialization failed"):
        _run(vllm_runner, monkeypatch, backend="ROCM_AITER_FA", model=GEMMA4)
    out, err = capfd.readouterr()
    assert "head_size not supported" in out + err


def test_check_catches_a_wrong_sliding_window(vllm_runner, monkeypatch):
    """Negative control: the comparison must fail when attention is wrong in
    a way only long prompts show (a 1024 window where the model has 2048)."""
    text = {**MUSE_SPARK.hf_overrides["text_config"], "sliding_window": 1024}
    wrong = replace(
        MUSE_SPARK, hf_overrides={**MUSE_SPARK.hf_overrides, "text_config": text}
    )
    ref = _reference(vllm_runner, monkeypatch, MUSE_SPARK, "auto")
    out, _ = _run(vllm_runner, monkeypatch, backend="ROCM_AITER_FA", model=wrong)
    assert _mean_prompt_logprob_diff(ref, out) > MUSE_SPARK.tolerance["auto"]


def test_shuffle_layout_refuses_sliding_window_model(vllm_runner, monkeypatch, capfd):
    """Shuffle decode has no window input: a sliding-window model must be
    refused at startup, not crash at its first decode."""
    monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1")
    with pytest.raises(Exception, match="initialization failed"):
        _run(vllm_runner, monkeypatch, backend="ROCM_AITER_FA", model=MUSE_SPARK)
    out, err = capfd.readouterr()
    assert "shuffle KV cache layout does not support sliding window" in out + err


@pytest.mark.slow_test
@pytest.mark.parametrize("cudagraph_mode, async_scheduling, model_runner_v2", FULL)
def test_engine_modes_full(
    vllm_runner, monkeypatch, cudagraph_mode, async_scheduling, model_runner_v2
):
    _compare(
        vllm_runner,
        monkeypatch,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


@pytest.mark.skipif(current_platform.device_count() < 2, reason="needs 2 GPUs")
def test_context_parallel_is_rejected(vllm_runner, monkeypatch, capfd):
    """The impl declares neither PCP nor DCP; asking for DCP with this backend
    must fail backend selection at startup, not produce wrong attention."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    # One KV head so DCP passes the TP > KV-heads check and reaches selection.
    with (
        pytest.raises(Exception, match="initialization failed"),
        vllm_runner(
            QWEN3.name,
            load_format="dummy",
            hf_overrides={**QWEN3.hf_overrides, "num_key_value_heads": 1},
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            attention_config={"backend": "ROCM_AITER_FA"},
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        ),
    ):
        pass
    # The reason is raised in the worker processes; read it from their output.
    out, err = capfd.readouterr()
    assert "DCP not supported" in out + err
