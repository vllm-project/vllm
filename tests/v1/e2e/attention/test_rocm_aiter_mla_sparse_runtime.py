# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA_SPARSE through the engine: scheduler, runners, graphs.

Each run builds a few-layer DSA model from a flagship architecture with dummy
weights (attention and indexer dims unchanged, everything else shrunk, index
top-k lowered to 256 so the test prompts are sparse) and compares its
logprobs with a reference run under the same engine settings:

- DeepSeek V3.2: every layer owns an indexer (128 heads, 576-wide latent).
- GLM-5.2: its repeating block, one indexer layer followed by three layers
  that reuse its top-k (64 heads, qk 192 + 64, v 256).
- GLM-5.3-Flash: three KDA layers and one rope-free DSA layer with the kpool
  indexer (64 heads, qk 256 + 0, v 256), at its real top-k of 2048.

ROCm has no second sparse-MLA backend, so the reference is the same backend
on AITER's Triton sparse MLA kernel (``VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA``,
gfx950) under the same engine settings. Both runs share the indexer, so they
select the same top-k (up to ties) and differ only in the attention kernels:
the default AITER ASM decode (with persistent work metadata), or vLLM's
Triton kernels for NoPE, against AITER's Triton kernel. A negative control
(a smaller top-k) must fail the check.

The batch mixes long prompts (chunked prefill -> sparse extends), a shared
prefix (prefix-cache hits), short prompts (dense prefill) and decode steps
under CUDA graph modes NONE / PIECEWISE / FULL_DECODE_ONLY /
FULL_AND_PIECEWISE, async scheduling on/off, model runner v1/v2, KV cache
dtype auto/fp8 and n-gram / MTP speculative decoding.
"""

from dataclasses import dataclass, field, replace
from functools import partial

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCM_AITER_MLA_SPARSE is ROCm only"
)

TOPK = 256


@dataclass(frozen=True)
class DummyModel:
    name: str
    hf_overrides: dict
    max_model_len: int = 4096
    long_words: int = 1200
    # LM-head gain: peaked enough to be informative (mean top logprob about
    # -2.5), not so peaked that every position saturates to logprob 0.
    logit_gain: float = 0.5
    # Mean prompt-logprob difference allowed vs the reference, per KV dtype.
    # Only rows longer than the top-k reach the sparse kernels in prefill, so
    # these are dominated by them. See each model for its calibration.
    tolerance: dict = field(default_factory=lambda: {"auto": 0.05, "fp8": 0.05})
    # Greedy samples are compared with check_logprobs_close unless the model
    # is too noisy for it even between two runs of the same kernel.
    check_samples: bool = True
    engine_kwargs: dict = field(default_factory=dict)


_SHRINK = {
    "hidden_size": 1024,
    "intermediate_size": 1024,
    "moe_intermediate_size": 256,
    # Every expert active: no routing ties for a kernel difference to flip.
    "n_routed_experts": 2,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "index_topk": TOPK,
}
# Calibrated on MI355: two runs of the reference differ by 0.007, the
# default path by 0.009 (bf16) / 0.011 (fp8); a top-k of 128 gives 0.25.
DSV32 = DummyModel(
    "deepseek-ai/DeepSeek-V3.2",
    {**_SHRINK, "num_hidden_layers": 2},
)
# Layer 0 owns the indexer; layers 1-3 reuse its top-k (the 4-layer block).
# Four layers of top-k selection amplify run-to-run noise (the indexer's
# top-k is not deterministic on ties): two reference runs differ by 0.27, the
# default path by 0.44, a top-k of 128 by 1.7 (bf16).
GLM52 = DummyModel(
    "zai-org/GLM-5.2",
    {
        **_SHRINK,
        "num_hidden_layers": 4,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 1,
    },
    logit_gain=2.0,
    tolerance={"auto": 0.8},
    # Greedy decode leaves the other run's top-5 even between two runs of
    # the reference; prompt logprobs carry the comparison.
    check_samples=False,
)
# GLM-5.3-Flash: its repeating block, three KDA layers then one rope-free
# DSA layer (64 heads, qk 256 + 0, v 256) with the kpool indexer. bf16 NoPE
# runs vLLM's Triton sparse kernels; dummy weights stay unquantized. Its fp8
# cache runs AITER's Triton kernel on both sides, so only bf16 is compared.
# Two reference runs differ by 0.003, the default path by 0.003, and the full
# context instead of the top-k by 0.6. Greedy decode of the long prompt
# differs between reference runs in different graph modes, so samples are
# not compared.
GLM53_FLASH = DummyModel(
    "zai-org/GLM-5.3-Flash",
    {
        "text_config": {
            **_SHRINK,
            # AITER's mHC kernels only take the real hidden sizes.
            "hidden_size": 4096,
            # The kpool top-k (index_topk / index_kpool) must be >= 512.
            "index_topk": 2048,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            "linear_attn_config": {
                "num_heads": 64,
                "gate_lower_bound": -5.0,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": [0, 1, 2],
                "full_attn_layers": [3],
            },
        },
        "vision_config": {"depth": 1},
        "quantization_config": None,
    },
    max_model_len=8192,
    long_words=2400,
    logit_gain=2.0,
    tolerance={"auto": 0.02},
    check_samples=False,
    engine_kwargs={"limit_mm_per_prompt": {"image": 0, "video": 0}},
)
MODELS = [
    pytest.param(DSV32, id="dsv32"),
    pytest.param(GLM52, id="glm52"),
    pytest.param(GLM53_FLASH, id="glm53_flash"),
]

MAX_TOKENS = 24
NUM_LOGPROBS = 5
SHARED_PREFIX = "You are a careful assistant. Answer briefly and cite sources. " * 8


def _prompts(model: DummyModel) -> list[str]:
    long = " ".join(f"item{i}" for i in range(model.long_words))
    return [
        "Hello",
        SHARED_PREFIX + "What is the capital of France?",
        SHARED_PREFIX + "What is the capital of Japan?",
        long,
        "one two three one two three one two three one two three",
        "The quick brown fox jumps over the lazy dog. " * 40,
    ]


def _shrink(overrides: tuple, hf_config):
    """Callable hf_overrides, so the MTP draft config is shrunk too (dict
    overrides only reach the target). Module level: it is pickled to the
    engine core."""
    for key, value in overrides:
        current = getattr(hf_config, key, None)
        if (
            isinstance(value, dict)
            and current is not None
            and not isinstance(current, dict)
        ):
            _shrink(tuple(value.items()), current)  # a nested sub-config
        else:
            setattr(hf_config, key, value)
    return hf_config


DUMMY_HALF_RANGE = 1e-3


def _scale_dummy_weights(model: torch.nn.Module, logit_gain: float) -> None:
    """Unit norm gains, matrices at std 1/sqrt(fan_in), a peaked LM head."""
    uniform_std = DUMMY_HALF_RANGE / 3**0.5
    for name, param in model.named_parameters():
        if param.dim() == 1 and "norm" in name:
            param.data.fill_(1.0)
        elif param.dim() >= 2 and param.dtype in (torch.bfloat16, torch.float16):
            gain = logit_gain if "lm_head" in name else 1.0
            param.data.mul_(gain * param.shape[-1] ** -0.5 / uniform_std)


def _sparse_impls(model: torch.nn.Module) -> list[str]:
    return sorted(
        {
            type(m.impl).__name__
            for m in model.modules()
            if getattr(m, "impl", None) is not None
            and getattr(m.impl, "is_sparse", False)
        }
    )


def _run(
    vllm_runner,
    monkeypatch,
    *,
    model=DSV32,
    triton_reference=False,
    kv_cache_dtype="auto",
    cudagraph_mode="NONE",
    async_scheduling=False,
    model_runner_v2=False,
    speculative=None,
):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv(
        "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", "1" if triton_reference else "0"
    )
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if model_runner_v2 else "0")
    kwargs = dict(model.engine_kwargs)
    if speculative == "ngram":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
        }
    elif speculative == "mtp":
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
    eager = cudagraph_mode == "NONE"
    with vllm_runner(
        model.name,
        load_format="dummy",
        hf_overrides=partial(_shrink, tuple(sorted(model.hf_overrides.items()))),
        dtype="bfloat16",
        max_model_len=model.max_model_len,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_dtype=kv_cache_dtype,
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
        return outputs, llm.apply_model(_sparse_impls)[0]


_REFERENCE: dict[tuple, list] = {}


def _mean_prompt_logprob_diff(ref: list, out: list) -> float:
    diffs = []
    for (*_, ref_prompt), (*_, out_prompt) in zip(ref, out):
        for ref_lp, out_lp in zip(ref_prompt[1:], out_prompt[1:]):
            token, logprob = next(iter(ref_lp.items()))
            diffs.append(abs(logprob.logprob - out_lp[token].logprob))
    return sum(diffs) / len(diffs)


def _reference(vllm_runner, monkeypatch, model, kv_cache_dtype, **kwargs):
    """The AITER Triton sparse MLA kernel under the same engine settings
    (graph mode, runner, async, spec decode), so only the kernels differ."""
    key = (model.name, kv_cache_dtype, tuple(sorted(kwargs.items())))
    if key not in _REFERENCE:
        _REFERENCE[key], _ = _run(
            vllm_runner,
            monkeypatch,
            model=model,
            triton_reference=True,
            kv_cache_dtype=kv_cache_dtype,
            **kwargs,
        )
    return _REFERENCE[key]


def _compare(vllm_runner, monkeypatch, model=DSV32, kv_cache_dtype="auto", **kwargs):
    if not current_platform.is_rocm():
        pytest.skip("ROCm only")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the AITER Triton sparse MLA reference is gfx950-only")
    ref = _reference(vllm_runner, monkeypatch, model, kv_cache_dtype, **kwargs)
    out, impls = _run(
        vllm_runner, monkeypatch, model=model, kv_cache_dtype=kv_cache_dtype, **kwargs
    )
    assert impls == ["ROCMAiterMLASparseImpl"]
    top = [max(v.logprob for v in lp.values()) for _, _, lps, _ in ref for lp in lps]
    assert sum(top) / len(top) > -6.0, "reference logits are nearly flat"
    diff = _mean_prompt_logprob_diff(ref, out)
    tol = model.tolerance[kv_cache_dtype]
    assert diff <= tol, f"mean prompt logprob diff {diff:.3f} > {tol}"
    if not model.check_samples:
        return
    check_logprobs_close(
        outputs_0_lst=ref,
        outputs_1_lst=out,
        name_0="aiter_triton_sparse_mla",
        name_1="ROCM_AITER_MLA_SPARSE",
    )


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
    vllm_runner, monkeypatch, model, cudagraph_mode, async_scheduling, model_runner_v2
):
    _compare(
        vllm_runner,
        monkeypatch,
        model=model,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


# GLM-5.2's run-to-run noise with an fp8 cache (0.6) is too close to its
# negative control (1.8) for a useful bound; DeepSeek V3.2 covers fp8.
@pytest.mark.parametrize("cudagraph_mode", ["NONE", "FULL_AND_PIECEWISE"])
def test_fp8_kv_cache(vllm_runner, monkeypatch, cudagraph_mode):
    _compare(
        vllm_runner,
        monkeypatch,
        model=DSV32,
        kv_cache_dtype="fp8",
        cudagraph_mode=cudagraph_mode,
    )


@pytest.mark.parametrize("speculative", ["ngram", "mtp"])
@pytest.mark.parametrize(
    "cudagraph_mode, model_runner_v2",
    [("NONE", False), ("FULL_AND_PIECEWISE", False), ("FULL_AND_PIECEWISE", True)],
)
def test_spec_decode(
    vllm_runner, monkeypatch, speculative, cudagraph_mode, model_runner_v2
):
    _compare(
        vllm_runner,
        monkeypatch,
        speculative=speculative,
        cudagraph_mode=cudagraph_mode,
        model_runner_v2=model_runner_v2,
        async_scheduling=model_runner_v2,
    )


def test_check_catches_a_wrong_topk(vllm_runner, monkeypatch):
    """Negative control: half the top-k must fail the comparison."""
    wrong = replace(DSV32, hf_overrides={**DSV32.hf_overrides, "index_topk": TOPK // 2})
    ref = _reference(vllm_runner, monkeypatch, DSV32, "auto")
    out, _ = _run(vllm_runner, monkeypatch, model=wrong)
    assert _mean_prompt_logprob_diff(ref, out) > DSV32.tolerance["auto"]


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
def test_decode_context_parallel_is_rejected(vllm_runner, monkeypatch, capfd):
    """The sparse impl has no DCP; asking for it must fail at startup."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    with (
        pytest.raises(Exception, match="initialization failed"),
        vllm_runner(
            DSV32.name,
            load_format="dummy",
            hf_overrides=partial(_shrink, tuple(sorted(DSV32.hf_overrides.items()))),
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        ),
    ):
        pass
    # The model pins its backend, so the selector's "DCP not supported" is
    # not reached; the runner's CP check refuses the impl instead.
    out, err = capfd.readouterr()
    assert "ROCMAiterMLASparseImpl does not" in out + err
