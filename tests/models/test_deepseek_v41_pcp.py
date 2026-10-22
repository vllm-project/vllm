# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    KernelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.models.deepseek_v41.amd import model as amd_model_module
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.nvidia import model as nvidia_model_module


@pytest.mark.parametrize(
    "model_module",
    [nvidia_model_module, amd_model_module],
    ids=["nvidia", "amd"],
)
@pytest.mark.parametrize(
    "pcp_size,use_ep", [(1, False), (2, True), (2, False), (4, False)]
)
@pytest.mark.parametrize("use_sp", [False, True])
def test_pcp_gathers_routing_ids_once_after_local_input_preparation(
    model_module, pcp_size, use_ep, use_sp, monkeypatch
):
    """Embedding keeps local IDs; all decoder layers receive aligned routing IDs."""
    gather = Mock(side_effect=lambda ids, dim: torch.cat([ids] * pcp_size, dim=dim))
    monkeypatch.setattr(
        model_module, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather)
    )
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(model_module, "sp_shard", lambda tensor: tensor[::2])
    monkeypatch.setattr(
        model_module, "sp_all_gather", lambda tensor: tensor.repeat_interleave(2, dim=0)
    )
    is_amd = model_module is amd_model_module
    if is_amd:
        monkeypatch.setattr(
            torch.ops.vllm, "hc_collapse_triton", lambda hidden, mix: hidden
        )
    else:
        monkeypatch.setattr(
            model_module, "hc_collapse_triton", lambda hidden, mix: hidden
        )
    embed = Mock(side_effect=lambda ids: ids[:, None].float())

    def run_layers_side_effect(layer_ids, hidden, *args):
        if is_amd:
            return hidden, None, None, None, hidden, [], None, False
        return hidden, None, None, None, hidden

    run_layers = Mock(side_effect=run_layers_side_effect)
    model = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size, enable_expert_parallel=use_ep
        ),
        use_native_mega_moe=False,
        use_mega_moe=False,
        use_sequence_parallel=use_sp,
        engram_hash=None,
        embed_input_ids=embed,
        start_layer=0,
        decoder_replay_start=2,
        decoder_replay_layers=None,
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=None,
        _run_layers=run_layers,
        _collapse=lambda hidden, *args: hidden,
        mhc_post=lambda hidden, *args: hidden,
        norm=lambda hidden: hidden,
    )
    for offset in (0, 10):
        local_ids = torch.tensor([3, 7, 11, 15]) + offset
        model_module.DeepseekV4Model.forward(model, local_ids, torch.arange(4), None)
        assert embed.call_args.args[0] is local_ids
        routing_ids = local_ids[::2] if use_sp else local_ids
        needs_gather = pcp_size > 1 and not use_ep
        if needs_gather:
            torch.testing.assert_close(gather.call_args.args[0], routing_ids)
            routing_ids = routing_ids.repeat(pcp_size)
        torch.testing.assert_close(run_layers.call_args.args[3], routing_ids)
    assert gather.call_count == 2 * int(needs_gather)


def _rocm_pcp_config() -> VllmConfig:
    return VllmConfig(
        cache_config=CacheConfig(enable_prefix_caching=False),
        parallel_config=ParallelConfig(prefill_context_parallel_size=2),
        kernel_config=KernelConfig(),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        device_config=DeviceConfig(device="cpu"),
    )


def test_rocm_pcp_accepts_the_eager_supported_scope() -> None:
    amd_model_module._validate_pcp_config(_rocm_pcp_config())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
        ("cache_config", "enable_prefix_caching", True),
        ("kernel_config", "moe_backend", "deep_gemm_mega_moe"),
        ("kernel_config", "monokernel", True),
        ("config", "speculative_config", object()),
        ("config", "kv_transfer_config", object()),
    ],
)
def test_rocm_pcp_rejects_unsupported_configurations(
    section: str, field: str, value: object
) -> None:
    config = _rocm_pcp_config()
    target = config if section == "config" else getattr(config, section)
    setattr(target, field, value)
    with pytest.raises(NotImplementedError, match="PP=DP=DCP=1"):
        amd_model_module._validate_pcp_config(config)


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_rocm_pcp_rejects_cuda_graph_modes(mode: CUDAGraphMode) -> None:
    config = _rocm_pcp_config()
    config.compilation_config.cudagraph_mode = mode
    with pytest.raises(NotImplementedError, match="eager execution"):
        amd_model_module._validate_pcp_config(config)


def test_rocm_pcp_backends_declare_support() -> None:
    assert DeepseekV41ROCMAiterMLASparseBackend.supports_pcp()
    assert DeepseekV41ROCMAiterSparseSWABackend.supports_pcp()


@pytest.mark.parametrize("parent_raises", [False, True])
def test_rocm_pcp_uses_parent_forward_and_restores_streams(
    monkeypatch: pytest.MonkeyPatch, parent_raises: bool
) -> None:
    attention = object.__new__(DeepseekV41ROCMAiterMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.use_pcp = True
    attention.indexer = object()
    attention._prepare_and_attn_fn = object()
    attention._prepare_and_attn_eager = object()
    aux_streams = [object()]
    attention.aux_stream_list = aux_streams
    parent_calls = 0

    def parent_forward(self, positions, hidden_states, llama_4_scaling=None):
        nonlocal parent_calls
        parent_calls += 1
        assert self.aux_stream_list is None
        if parent_raises:
            raise RuntimeError("parent failure")
        return hidden_states

    def unexpected_parallel_path(*args, **kwargs):
        raise AssertionError("PCP must not enter the ROCm parallel attention path")

    execute_in_parallel = Mock()
    monkeypatch.setattr(DeepseekV4Attention, "forward", parent_forward)
    monkeypatch.setattr(
        "vllm.models.deepseek_v41.amd.rocm.execute_in_parallel",
        execute_in_parallel,
    )
    monkeypatch.setattr(
        DeepseekV41ROCMAiterMLAAttention,
        "_forward_csa2_full",
        unexpected_parallel_path,
    )
    monkeypatch.setattr(
        DeepseekV41ROCMAiterMLAAttention,
        "_forward_csa2_reindex",
        unexpected_parallel_path,
    )

    positions = torch.arange(2)
    hidden_states = torch.ones(2, 4)
    if parent_raises:
        with pytest.raises(RuntimeError, match="parent failure"):
            attention(positions, hidden_states)
    else:
        assert attention(positions, hidden_states) is hidden_states
    assert parent_calls == 1
    execute_in_parallel.assert_not_called()
    assert attention.aux_stream_list is aux_streams
