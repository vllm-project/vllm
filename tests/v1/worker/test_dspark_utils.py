# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)


@dataclass
class _FakeHFConfig:
    model_type: str


@dataclass
class _FakeModelConfig:
    hf_config: _FakeHFConfig


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
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


@pytest.mark.parametrize(
    ("model_type", "enable_eplb", "num_redundant_experts", "shares_eplb_config"),
    [
        ("deepseek_v4", True, 32, True),
        ("deepseek_v41", False, 0, False),
    ],
)
def test_dspark_parallel_config_eplb_by_model_type(
    model_type: str,
    enable_eplb: bool,
    num_redundant_experts: int,
    shares_eplb_config: bool,
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
    assert draft_config.enable_eplb is enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == num_redundant_experts
    assert not draft_config.enable_elastic_ep
    if shares_eplb_config:
        assert draft_config.eplb_config is target_config.eplb_config
    else:
        assert draft_config.eplb_config is not target_config.eplb_config


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
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)
