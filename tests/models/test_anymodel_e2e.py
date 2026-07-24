# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E initialization tests for AnyModel.

Downloads the real HF config (not weights), injects AnyModel overrides,
and initializes with load_format="dummy". Requires GPU and network access.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ..utils import create_new_process_for_each_test
from .utils import dummy_hf_overrides

_NUM_LAYERS = 4


def _dense_per_layer_config(hf_config: PretrainedConfig) -> dict[int, dict]:
    text_cfg = hf_config.get_text_config()
    isize = getattr(text_cfg, "intermediate_size", 1024)
    kv = getattr(
        text_cfg,
        "num_key_value_heads",
        getattr(text_cfg, "num_attention_heads", 4),
    )
    return {
        1: {"intermediate_size": max(64, isize // 2)},
        2: {"num_key_value_heads": max(1, kv // 2)},
        3: {"skip": ["attention"], "intermediate_size": max(64, isize // 2)},
    }


def _moe_per_layer_config(
    hf_config: PretrainedConfig,
    *,
    experts_field: str = "num_local_experts",
    expert_size_field: str | None = None,
) -> dict[int, dict]:
    text_cfg = hf_config.get_text_config()
    n_experts = getattr(text_cfg, experts_field, 4)
    exp_isize = (
        getattr(text_cfg, expert_size_field, None) if expert_size_field else None
    )
    reduced = max(2, n_experts // 2)
    moe_override: dict[str, Any] = {experts_field: reduced}
    if exp_isize is not None and expert_size_field is not None:
        moe_override[expert_size_field] = max(64, exp_isize // 2)
    return {
        1: moe_override,
        3: {"skip": ["attention"]},
    }


def _nemotronh_per_layer_config(hf_config: PretrainedConfig) -> dict[int, dict]:
    kv = getattr(hf_config, "num_key_value_heads", 4)
    return {
        2: {"num_key_value_heads": max(1, kv // 2)},
        3: {"skip": ["mlp"]},
    }


@dataclass
class _Case:
    id: str
    base_arch: str
    hf_model: str
    make_per_layer_config: Any
    trust_remote_code: bool = False
    max_model_len: int = 1024
    attention_backend: str | None = None
    extra_hf_overrides: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return self.id


_CASES: list[_Case] = [
    # Dense
    _Case(
        "llama",
        "LlamaForCausalLM",
        "meta-llama/Llama-3.2-1B-Instruct",
        _dense_per_layer_config,
    ),
    _Case(
        "mistral",
        "MistralForCausalLM",
        "mistralai/Mistral-7B-Instruct-v0.1",
        _dense_per_layer_config,
    ),
    _Case(
        "qwen2",
        "Qwen2ForCausalLM",
        "Qwen/Qwen2-0.5B-Instruct",
        _dense_per_layer_config,
    ),
    _Case("qwen3", "Qwen3ForCausalLM", "Qwen/Qwen3-8B", _dense_per_layer_config),
    # MoE
    _Case(
        "qwen2moe",
        "Qwen2MoeForCausalLM",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        partial(
            _moe_per_layer_config,
            experts_field="num_experts",
            expert_size_field="moe_intermediate_size",
        ),
    ),
    _Case(
        "mixtral",
        "MixtralForCausalLM",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        partial(_moe_per_layer_config, experts_field="num_local_experts"),
    ),
    _Case(
        "gptoss",
        "GptOssForCausalLM",
        "lmsys/gpt-oss-20b-bf16",
        partial(_moe_per_layer_config, experts_field="num_local_experts"),
        attention_backend="TRITON_ATTN",
    ),
    # Hybrid
    _Case(
        "nemotronh",
        "NemotronHForCausalLM",
        "nvidia/Nemotron-H-8B-Base-8K",
        _nemotronh_per_layer_config,
        trust_remote_code=True,
        extra_hf_overrides={"hybrid_override_pattern": "*-*-"},
    ),
    # Multimodal
    _Case(
        "qwen3vl",
        "Qwen3VLForConditionalGeneration",
        "Qwen/Qwen3-VL-4B-Instruct",
        _dense_per_layer_config,
        max_model_len=4096,
    ),
]


def _anymodel_hf_overrides(
    hf_config: PretrainedConfig,
    *,
    case: _Case,
) -> PretrainedConfig:
    hf_config = dummy_hf_overrides(hf_config, model_arch=case.base_arch)
    text_cfg = hf_config.get_text_config()
    text_cfg.num_hidden_layers = _NUM_LAYERS
    hf_config.architectures = ["AnyModel"]
    hf_config.base_architecture = case.base_arch
    for key, val in case.extra_hf_overrides.items():
        setattr(text_cfg, key, val)
    text_cfg.per_layer_config = case.make_per_layer_config(hf_config)
    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.update({"num_layers": 1, "num_hidden_layers": 1})
    return hf_config


def _kv_cache_stub(self, vllm_config):
    kv_cache_specs = self.model_executor.get_kv_cache_specs()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,
        [10 * GiB_bytes] * len(kv_cache_specs),
    )
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
    if kv_cache_groups:
        vllm_config.cache_config.block_size = min(
            g.kv_cache_spec.block_size for g in kv_cache_groups
        )
    self.model_executor.initialize_from_config(kv_cache_configs)
    return scheduler_kv_cache_config


def _get_model(llm: LLM):
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model


def _teardown_engine():
    """Release an engine before creating another in the same process.

    Beyond the usual dist/memory cleanup, drop the cached CUDA-graph memory
    pool. The pool handle is a class-level attribute on the platform; reusing
    it for the next engine's graph capture triggers a `use_count > 0` assert
    in the caching allocator once the previous engine has been torn down.
    Scoped to these tests, which are the only ones creating two
    cudagraph-capturing engines back-to-back in one process.
    """
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
    from vllm.platforms import current_platform

    current_platform.__class__._global_graph_pool = None


@create_new_process_for_each_test()
def _run_anymodel_e2e(case: _Case):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    attention_config = (
        {"backend": case.attention_backend} if case.attention_backend else None
    )
    with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
        LLM(
            case.hf_model,
            load_format="dummy",
            hf_overrides=partial(_anymodel_hf_overrides, case=case),
            trust_remote_code=case.trust_remote_code,
            max_model_len=case.max_model_len,
            enforce_eager=False,
            gpu_memory_utilization=0.80,
            attention_config=attention_config,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.id)
def test_anymodel_e2e(case: _Case):
    _run_anymodel_e2e(case)


_PARITY_MODEL = "Qwen/Qwen2-0.5B-Instruct"
_PARITY_BASE_ARCH = "Qwen2ForCausalLM"


def _identity_anymodel_overrides(
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    text_cfg = hf_config.get_text_config()
    hf_config.architectures = ["AnyModel"]
    hf_config.base_architecture = _PARITY_BASE_ARCH
    # Sparse dict: empty means every layer uses the global config unchanged.
    text_cfg.per_layer_config = {}
    return hf_config


@create_new_process_for_each_test()
def _run_anymodel_parity():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sampling = SamplingParams(temperature=0, max_tokens=64)
    prompt = "The capital of France is"

    base_llm = LLM(_PARITY_MODEL, enforce_eager=False, gpu_memory_utilization=0.4)
    base_text = base_llm.generate([prompt], sampling)[0].outputs[0].text
    del base_llm
    _teardown_engine()

    anymodel_llm = LLM(
        _PARITY_MODEL,
        hf_overrides=_identity_anymodel_overrides,
        enforce_eager=False,
        gpu_memory_utilization=0.4,
    )
    anymodel_text = anymodel_llm.generate([prompt], sampling)[0].outputs[0].text
    assert base_text == anymodel_text, (
        f"Parity mismatch:\n  base:     {base_text!r}\n  anymodel: {anymodel_text!r}"
    )


def test_anymodel_weight_loading_parity():
    _run_anymodel_parity()


def _find_decoder_layers(model) -> list:
    """Walk common paths to the decoder ModuleList (layers / model.layers /
    language_model.model.layers)."""
    for path in ("model.layers", "language_model.model.layers", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__iter__") and hasattr(obj, "__len__"):
            return list(obj)
    raise AssertionError("Could not locate decoder layers on model")


def _capture_layer_outputs(model) -> tuple[list, list]:
    """Register forward hooks capturing each decoder layer's hidden_states output.

    Returns (hidden_states_per_layer, hook_handles). Hooks should be removed
    after the forward pass. Layers are captured by index in model.layers order.
    """
    layers = _find_decoder_layers(model)
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def _make_hook(idx):
        def _hook(module, args, output):
            # Decoder layers typically return hidden_states (sometimes a tuple
            # (hidden_states, residual)). Grab the first tensor and detach to cpu.
            if isinstance(output, tuple):
                hs = next((o for o in output if isinstance(o, torch.Tensor)), None)
            else:
                hs = output if isinstance(output, torch.Tensor) else None
            if hs is not None:
                captured[idx] = hs.detach().to("cpu")

        return _hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_hook(i)))
    return captured, handles


@create_new_process_for_each_test()
def _run_anymodel_layer_parity():
    """Assert per-decoder-layer hidden states match exactly between plain and
    identity-wrapped AnyModel. With identical weights and no block overrides,
    every layer's output tensor should be bitwise equal; any drift would
    indicate the wrapper is perturbing the forward path."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sampling = SamplingParams(temperature=0, max_tokens=8)
    prompt = "The capital of France is"
    common = dict(enforce_eager=True, gpu_memory_utilization=0.4)

    base_llm = LLM(_PARITY_MODEL, **common)
    base_captured, base_handles = _capture_layer_outputs(_get_model(base_llm))
    base_out = base_llm.generate([prompt], sampling)[0]
    for h in base_handles:
        h.remove()
    base_ids = tuple(base_out.outputs[0].token_ids)
    base_layers = [t for t in base_captured if t is not None]
    assert len(base_layers) > 0, "No layer outputs captured for base model"
    del base_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    any_llm = LLM(
        _PARITY_MODEL,
        hf_overrides=_identity_anymodel_overrides,
        **common,
    )
    any_captured, any_handles = _capture_layer_outputs(_get_model(any_llm))
    any_out = any_llm.generate([prompt], sampling)[0]
    for h in any_handles:
        h.remove()
    any_ids = tuple(any_out.outputs[0].token_ids)
    any_layers = [t for t in any_captured if t is not None]

    assert base_ids == any_ids, (
        f"Token-id parity mismatch:\n  base:     {base_ids}\n  anymodel: {any_ids}"
    )
    assert len(base_layers) == len(any_layers), (
        f"Layer count mismatch: base={len(base_layers)} vs anymodel={len(any_layers)}"
    )
    for i, (b, a) in enumerate(zip(base_layers, any_layers)):
        assert b.shape == a.shape, f"Layer {i} shape mismatch: {b.shape} vs {a.shape}"
        # With identical weights, identical input, and no overrides, every
        # kernel path is the same — assert bitwise equality.
        assert torch.equal(b, a), (
            f"Layer {i} hidden-state divergence "
            f"(max_abs_diff={(b - a).abs().max().item():.3e})"
        )


def test_anymodel_layer_parity():
    """Per-layer hidden-state equality for identity-wrapped AnyModel."""
    _run_anymodel_layer_parity()


@create_new_process_for_each_test()
def _run_anymodel_throughput_parity():
    """Assert identity-wrapped AnyModel throughput is not materially worse than
    plain. Regressions here catch issues like the pooling-runner misclassification
    that silently halved max_cudagraph_capture_size."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sampling = SamplingParams(temperature=0, max_tokens=64, seed=0)
    # Enough prompts to amortize startup / warmup cost.
    prompts = [f"Explain in one sentence what number {i} means." for i in range(32)]
    common = dict(enforce_eager=False, gpu_memory_utilization=0.4)

    def _time_generate(llm):
        # Warm-up to trigger any lazy compile/capture work outside the timed run.
        llm.generate(prompts[:4], sampling)
        start = time.perf_counter()
        outs = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - start
        n_out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        return elapsed, n_out_tokens, [tuple(o.outputs[0].token_ids) for o in outs]

    base_llm = LLM(_PARITY_MODEL, **common)
    base_time, base_tokens, base_ids = _time_generate(base_llm)
    del base_llm
    _teardown_engine()

    any_llm = LLM(_PARITY_MODEL, hf_overrides=_identity_anymodel_overrides, **common)
    any_time, any_tokens, any_ids = _time_generate(any_llm)

    assert base_ids == any_ids, (
        "Generated token parity mismatch between base and AnyModel"
    )
    assert base_tokens == any_tokens

    base_tps = base_tokens / base_time
    any_tps = any_tokens / any_time
    # AnyModel is allowed a narrow overhead band. The broken state (pooling
    # misclassification) was ~-36%; anything within 10% is fine noise-wise.
    ratio = any_tps / base_tps
    assert ratio > 0.90, (
        f"AnyModel throughput regressed >10%: "
        f"base={base_tps:.1f} tok/s vs anymodel={any_tps:.1f} tok/s (ratio={ratio:.2f})"
    )


def test_anymodel_throughput_parity():
    """AnyModel (identity wrap) throughput must stay within ~10% of plain."""
    _run_anymodel_throughput_parity()


_REDUCTION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_REDUCTION_ARCH = "LlamaForCausalLM"


def _base_llama_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = dummy_hf_overrides(hf_config, model_arch=_REDUCTION_ARCH)
    hf_config.get_text_config().num_hidden_layers = _NUM_LAYERS
    return hf_config


def _reduced_anymodel_overrides(
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    hf_config = dummy_hf_overrides(hf_config, model_arch=_REDUCTION_ARCH)
    text_cfg = hf_config.get_text_config()
    text_cfg.num_hidden_layers = _NUM_LAYERS
    hf_config.architectures = ["AnyModel"]
    hf_config.base_architecture = _REDUCTION_ARCH
    isize = getattr(text_cfg, "intermediate_size", 1024)
    kv = getattr(
        text_cfg,
        "num_key_value_heads",
        getattr(text_cfg, "num_attention_heads", 4),
    )
    half_i, half_kv = max(64, isize // 2), max(1, kv // 2)
    text_cfg.per_layer_config = {
        0: {"intermediate_size": half_i},
        1: {"num_key_value_heads": half_kv, "intermediate_size": half_i},
        2: {"skip": ["attention"], "intermediate_size": half_i},
        3: {"skip": ["attention", "mlp"]},
    }
    return hf_config


@create_new_process_for_each_test()
def _run_anymodel_size_reduction():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    common = dict(load_format="dummy", enforce_eager=True, gpu_memory_utilization=0.80)
    with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
        base_llm = LLM(_REDUCTION_MODEL, hf_overrides=_base_llama_overrides, **common)
    base_params = sum(p.numel() for p in _get_model(base_llm).parameters())
    del base_llm
    cleanup_dist_env_and_memory()

    with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
        any_llm = LLM(
            _REDUCTION_MODEL, hf_overrides=_reduced_anymodel_overrides, **common
        )
    any_params = sum(p.numel() for p in _get_model(any_llm).parameters())
    assert any_params < base_params, (
        f"AnyModel should have fewer params: {any_params:,} >= {base_params:,}"
    )


def test_anymodel_size_reduction():
    _run_anymodel_size_reduction()


_NAS_CONFIG_PATH = Path(__file__).resolve().parents[0] / "fixtures" / "nas_config.json"


@create_new_process_for_each_test()
def _run_puzzletron_nas_config():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm.model_executor.models.anymodel import (
        NoOpAttention,
        NoOpMLP,
        NoOpNorm,
    )

    with open(_NAS_CONFIG_PATH) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(config))
        with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
            llm = LLM(
                tmp,
                tokenizer="meta-llama/Llama-3.2-1B-Instruct",
                load_format="dummy",
                enforce_eager=True,
                max_model_len=1024,
                gpu_memory_utilization=0.90,
            )

    model = _get_model(llm)
    layers = model.model.layers
    entries = config["per_layer_config"]
    assert len(layers) == config["num_hidden_layers"]

    hs = config["hidden_size"]
    hd = config["head_dim"]
    nh = config["num_attention_heads"]
    gkv = config["num_key_value_heads"]
    gisize = config["intermediate_size"]

    for i, layer in enumerate(layers):
        # JSON keys are strings; fall back to {} for layers not in the sparse dict.
        entry = entries.get(str(i), entries.get(i, {}))
        skip = set(entry.get("skip") or ())
        a_noop = "attention" in skip
        f_noop = "mlp" in skip

        assert isinstance(layer.self_attn, NoOpAttention) == a_noop, f"L{i} attn"
        assert isinstance(layer.input_layernorm, NoOpNorm) == a_noop, f"L{i} norm"
        assert isinstance(layer.mlp, NoOpMLP) == f_noop, f"L{i} ffn"
        assert isinstance(layer.post_attention_layernorm, NoOpNorm) == f_noop, (
            f"L{i} ffn norm"
        )

        if not a_noop:
            kv = entry.get("num_key_value_heads", gkv)
            assert layer.self_attn.qkv_proj.weight.shape == (
                (nh + 2 * kv) * hd,
                hs,
            ), f"L{i} qkv"

        if not f_noop:
            isize = entry.get("intermediate_size", gisize)
            assert layer.mlp.down_proj.weight.shape == (hs, isize), f"L{i} down"
            assert layer.mlp.gate_up_proj.weight.shape == (
                2 * isize,
                hs,
            ), f"L{i} gate_up"


def test_puzzletron_nas_config():
    _run_puzzletron_nas_config()
