# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.config import LoRAConfig, ParallelConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _get_dspark_parallel_config,
    load_dspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


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
            draft_model_config=SimpleNamespace(use_mla=use_mla)
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


@dataclass
class _FakeDraftHFConfig:
    model_type: str
    # dflash_has_any_non_causal() walks the draft's layers.
    num_hidden_layers: int = 0


@dataclass
class _FakeAttentionConfig:
    backend: object = None
    use_non_causal: bool = False


@dataclass
class _FakeSpeculativeConfig:
    draft_model_config: _FakeModelConfig
    draft_parallel_config: SimpleNamespace
    attention_backend: object = None
    kv_cache_dtype: str | None = None


@dataclass
class _FakeVllmConfig:
    model_config: _FakeModelConfig
    parallel_config: _FakeParallelConfig
    attention_config: _FakeAttentionConfig
    speculative_config: _FakeSpeculativeConfig
    lora_config: LoRAConfig | None
    cache_config: SimpleNamespace = field(default_factory=SimpleNamespace)
    load_config: SimpleNamespace = field(default_factory=SimpleNamespace)
    quant_config: object = None


def test_dspark_drafter_is_built_without_the_targets_lora_config(monkeypatch):
    """The drafter never gets a LoRA context, so it must not be LoRA-aware.

    With the target's lora_config in place, the drafter's FusedMoE layers are
    built with is_lora_enabled=True and the MoE oracle picks LoRA-aware experts
    (e.g. TrtLlmMxfp4LoRAExperts under moe_backend=flashinfer_trtllm). The
    drafter is not wrapped by the LoRA manager, so its first forward pass dies
    on `assert lora_context is not None`.
    """
    import vllm.model_executor.model_loader as model_loader
    import vllm.model_executor.models.utils as model_utils
    import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils

    draft_model_config = _FakeModelConfig(
        hf_config=_FakeDraftHFConfig(model_type="deepseek_v41")
    )
    vllm_config = _FakeVllmConfig(
        model_config=draft_model_config,
        parallel_config=_FakeParallelConfig(
            enable_eplb=False,
            eplb_config=_FakeEPLBConfig(num_redundant_experts=0),
            enable_elastic_ep=False,
        ),
        attention_config=_FakeAttentionConfig(),
        speculative_config=_FakeSpeculativeConfig(
            draft_model_config=draft_model_config,
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=4),
        ),
        lora_config=LoRAConfig(max_loras=2),
    )

    class _Captured(Exception):
        """Stops the load once we hold the config it would have used."""

    def _capture(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    monkeypatch.setattr(model_loader, "get_model", _capture)
    monkeypatch.setattr(model_utils, "get_draft_quant_config", lambda cfg: None)
    monkeypatch.setattr(
        dspark_utils, "get_draft_load_config", lambda cfg: cfg.load_config
    )
    monkeypatch.setattr(
        dspark_utils, "get_pp_safe_draft_load_config", lambda load_config: load_config
    )

    with pytest.raises(_Captured) as captured:
        load_dspark_model(object(), vllm_config)

    assert captured.value.args[0].lora_config is None
    # The target keeps its own adapters.
    assert vllm_config.lora_config is not None
