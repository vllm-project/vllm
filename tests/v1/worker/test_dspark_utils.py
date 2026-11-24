# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.v1.core.utils import full_layer
from vllm.config import ParallelConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)


@dataclass(slots=True)
class _FakeHFConfig:
    model_type: str


@dataclass(slots=True)
class _FakeModelConfig:
    hf_config: _FakeHFConfig


@dataclass(slots=True)
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass(slots=True)
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="DSV4/DSV41 EPLB is disabled on ROCm",
)
def test_dspark_parallel_config_preserves_eplb_for_dsv4_and_dsv41(
    model_type: str,
):
    target_config = _FakeParallelConfig()
    draft_model_config = _FakeModelConfig(
        hf_config=_FakeHFConfig(model_type=model_type)
    )

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
        draft_model_config=draft_model_config,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert draft_config.enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == 32
    assert not draft_config.enable_elastic_ep
    assert draft_config.eplb_config is target_config.eplb_config


def test_dspark_parallel_config_can_disable_draft_eplb():
    target_config = _FakeParallelConfig()
    draft_model_config = _FakeModelConfig(
        hf_config=_FakeHFConfig(model_type="deepseek_v41")
    )

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
        draft_model_config=draft_model_config,
        draft_enable_eplb=False,
    )

    assert not draft_config.enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == 0
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_decode_context_parallel_size=dcp_size if use_mla else 1
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize("use_mla", [False, True])
@pytest.mark.parametrize(
    "window_config",
    [
        {},
        {"sliding_window": 2051},
        # Windowed layers resolved from the DFlash config alone.
        {"dflash_config": {"use_swa": True, "swa_window_size": 2051}},
        {"layer_types": ["full_attention", "sliding_attention"]},
    ],
)
def test_default_draft_decode_context_parallel_size(dflash, use_mla, window_config):
    spec = SimpleNamespace(
        target_parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        draft_model_config=SimpleNamespace(
            use_mla=use_mla,
            hf_config=SimpleNamespace(**window_config),
            get_sliding_window=lambda: window_config.get("sliding_window"),
        ),
    )
    config_cls = DFlashSpeculatorConfig if dflash else DraftModelSpeculatorConfig
    # Only full-attention MLA DFlash drafts shard their cache with the target.
    expected = 1 if dflash and (not use_mla or window_config) else 4
    assert config_cls.default_draft_decode_context_parallel_size(spec) == expected


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_dflash_rejects_windowed_draft_layers_under_dcp(monkeypatch, draft_dcp_size):
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module

    layers = {
        "draft.0.attn": SimpleNamespace(sliding_window=None),
        "draft.1.attn": SimpleNamespace(sliding_window=128),
    }
    monkeypatch.setattr(DraftModelSpeculator, "load_model", lambda self, _: None)
    monkeypatch.setattr(
        dflash_module,
        "get_layers_from_vllm_config",
        lambda _config, _type, names: {name: layers[name] for name in names},
    )
    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = None
    speculator.dcp_size = draft_dcp_size
    speculator.draft_attn_layer_names = set(layers)

    if draft_dcp_size == 1:
        speculator.load_model(None)
    else:
        with pytest.raises(ValueError, match=r"\['draft.1.attn'\]"):
            speculator.load_model(None)


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase, AttentionMetadata

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl: AttentionImplBase[AttentionMetadata] = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    layer = full_layer(
        AttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    )
    monkeypatch.setattr(
        model_runner,
        "get_kv_cache_layer_configs",
        lambda _: {"target": layer, "draft": layer},
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    layer_configs = runner.get_kv_cache_layer_configs()

    assert layer_configs["target"].spec.dcp_sharded
    assert layer_configs["draft"].spec.dcp_sharded == (draft_dcp_size == 4)


def _run_load_dspark_model_capturing_draft_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_model_path: str,
    draft_model_path: str,
    draft_architectures: list[str],
    draft_model_type: str,
    target_quantization: str | None,
    draft_quantization: str | None = None,
    spec_quantization: str | None = None,
    target_quant_config: object | None = None,
    resolved_by_name: dict[str, object] | None = None,
) -> tuple[Any, Any]:
    from contextlib import nullcontext

    from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils

    draft_hf_config = SimpleNamespace(
        model_type=draft_model_type,
        draft_vocab_size=None,
    )
    draft_model_config = SimpleNamespace(
        model=draft_model_path,
        architectures=draft_architectures,
        quantization=draft_quantization,
        hf_config=draft_hf_config,
        get_vocab_size=lambda: 128,
    )
    target_model_config = SimpleNamespace(
        model=target_model_path,
        quantization=target_quantization,
        get_vocab_size=lambda: 128,
    )
    speculative_config = SimpleNamespace(
        quantization=spec_quantization,
        draft_model_config=draft_model_config,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
        draft_enable_eplb=False,
        attention_backend=None,
        kv_cache_dtype=None,
        draft_lm_head_quantization=None,
        apply_draft_overrides=lambda cfg: cfg,
    )
    vllm_config = SimpleNamespace(
        model_config=target_model_config,
        speculative_config=speculative_config,
        parallel_config=ParallelConfig(),
        attention_config=SimpleNamespace(backend=None, use_non_causal=False),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        load_config=SimpleNamespace(load_format="auto"),
        quant_config=target_quant_config,
    )

    def _fake_replace(config, **changes):
        data = vars(config).copy()
        data.update(changes)
        return SimpleNamespace(**data)

    captured: dict[str, Any] = {}
    draft_nn = torch.nn.Module()
    draft_nn.model = torch.nn.Module()
    target_nn = torch.nn.Module()
    target_nn.model = torch.nn.Module()

    def _fake_get_model(*, vllm_config, model_config):
        captured["draft_vllm_config"] = vllm_config
        captured["draft_model_config"] = model_config
        return draft_nn

    def _fake_get_draft_quant_config(cfg):
        quant_name = cfg.speculative_config.draft_model_config.quantization
        if resolved_by_name is not None and quant_name is not None:
            return resolved_by_name.get(quant_name)
        return None

    monkeypatch.setattr(dspark_utils, "replace", _fake_replace)
    monkeypatch.setattr(
        dspark_utils, "get_draft_load_config", lambda cfg: cfg.load_config
    )
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag", lambda _tag: nullcontext()
    )
    monkeypatch.setattr("vllm.model_executor.model_loader.get_model", _fake_get_model)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _hf: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_draft_quant_config",
        _fake_get_draft_quant_config,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_target_lm_head",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.eagle.utils.maybe_share_target_embed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.eagle.utils.share_target_lm_head",
        lambda *_args, **_kwargs: None,
    )

    loaded = dspark_utils.load_dspark_model(target_nn, vllm_config)
    assert loaded is draft_nn
    return captured["draft_vllm_config"], captured["draft_model_config"]


@pytest.mark.parametrize(
    ("draft_arch", "draft_model_type"),
    [
        ("DSparkDraftModel", "deepseek_v4"),
        ("DSparkV41DraftModel", "deepseek_v41"),
    ],
)
def test_load_dspark_model_inherits_late_resolved_target_quant_config(
    monkeypatch: pytest.MonkeyPatch,
    draft_arch: str,
    draft_model_type: str,
) -> None:
    """Same-checkpoint DSpark drafts inherit quantization resolved after spec init."""
    target_quant_obj = object()
    draft_quant_obj = object()

    draft_vllm_cfg, draft_model_cfg = _run_load_dspark_model_capturing_draft_config(
        monkeypatch,
        target_model_path="/models/dsv4-quant",
        draft_model_path="/models/dsv4-quant",
        draft_architectures=[draft_arch],
        draft_model_type=draft_model_type,
        target_quantization="ascend",
        draft_quantization=None,
        target_quant_config=target_quant_obj,
        resolved_by_name={"ascend": draft_quant_obj},
    )
    assert draft_model_cfg.quantization == "ascend"
    assert draft_vllm_cfg.quant_config is draft_quant_obj

    # Fallback when the platform populated vllm_config.quant_config directly.
    draft_vllm_cfg2, _ = _run_load_dspark_model_capturing_draft_config(
        monkeypatch,
        target_model_path="/models/dsv4-quant",
        draft_model_path="/models/dsv4-quant",
        draft_architectures=[draft_arch],
        draft_model_type=draft_model_type,
        target_quantization=None,
        draft_quantization=None,
        target_quant_config=target_quant_obj,
    )
    assert draft_vllm_cfg2.quant_config is target_quant_obj


def test_load_dspark_model_keeps_self_contained_and_ml4_drafts_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-contained and Mistral Large 4 BF16 drafts do not inherit target quant."""
    from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

    target_quant_obj = object()

    # Separate self-contained DSpark checkpoint remains unquantized.
    sep_vllm_cfg, sep_model_cfg = _run_load_dspark_model_capturing_draft_config(
        monkeypatch,
        target_model_path="/models/qwen3-fp8",
        draft_model_path="/models/dspark-qwen3-bf16",
        draft_architectures=["Qwen3DSparkModel"],
        draft_model_type="qwen3",
        target_quantization="fp8",
        target_quant_config=target_quant_obj,
        resolved_by_name={"fp8": object()},
    )
    assert sep_model_cfg.quantization is None
    assert sep_vllm_cfg.quant_config is None

    # Mistral Large 4 in-checkpoint BF16 draft remains unquantized.
    ml4_vllm_cfg, ml4_model_cfg = _run_load_dspark_model_capturing_draft_config(
        monkeypatch,
        target_model_path="/models/mistral-large-4-fp8",
        draft_model_path="/models/mistral-large-4-fp8",
        draft_architectures=["MistralLarge4DSparkModel"],
        draft_model_type="mistral_large_4",
        target_quantization="fp8",
        target_quant_config=target_quant_obj,
        resolved_by_name={"fp8": object()},
    )
    assert ml4_model_cfg.quantization is None
    assert ml4_vllm_cfg.quant_config is None

    # Explicit speculative_config.quantization is preserved by adapt_draft_model_config.
    draft = SimpleNamespace(
        speculator_config_class=None,
        hf_config=SimpleNamespace(model_type="deepseek_mtp", architectures=[]),
        quantization="compressed-tensors",
    )
    spec = SimpleNamespace(
        quantization="compressed-tensors",
        draft_model_config=draft,
        target_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4"),
            quantization="fp8",
        ),
        update_arch_=lambda: None,
    )
    DSparkSpeculatorConfig.adapt_draft_model_config(spec)
    assert draft.quantization == "compressed-tensors"
