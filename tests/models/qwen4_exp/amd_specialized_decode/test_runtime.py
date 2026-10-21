# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of --monokernel for Qwen3.8-Flash-Next: the config refusals
(every reason in one error), the widths, and the per-step routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.qwen4_exp.amd.specialized_decode import runtime
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.routing import step_reason
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="the Flash-Next specialized decode is ROCm only",
)


def _text_config(**over):
    values = dict(zip(L.SHAPE_NAMES, L.SHAPES))
    values.update(norm_topk_prob=True, num_hidden_layers=40)
    values.update(over)
    return SimpleNamespace(**values)


def _vllm_config(monokernel=True, **parallel):
    values = dict(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_ubatching=False,
        enable_hc_sp=False,
        use_sequence_parallel_moe=False,
    )
    values.update(parallel)
    return SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        parallel_config=SimpleNamespace(**values),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL
        ),
        lora_config=None,
        kv_transfer_config=None,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        speculative_config=None,
        weight_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
            hf_text_config=_text_config(),
        ),
    )


@pytest.fixture
def supported_platform(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    monkeypatch.setattr(runtime, "platform_reasons", lambda: [])


def test_off_without_monokernel():
    # nothing else is read: every deployment runs vLLM's decode
    config = SimpleNamespace(kernel_config=SimpleNamespace(monokernel=False))
    assert runtime.specialized_decode_config(config) is None


def test_supported_config(supported_platform):
    config = runtime.specialized_decode_config(_vllm_config())
    assert config == runtime.DecodeConfig()
    assert config.widths == tuple(range(1, 17))


def test_refusals_report_every_reason(supported_platform, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0")
    config = _vllm_config(tensor_parallel_size=4, enable_expert_parallel=True)
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_ssm_cache_dtype = "auto"
    config.speculative_config = SimpleNamespace(method="mtp")
    config.lora_config = SimpleNamespace()
    config.use_v2_model_runner = False
    config.model_config.hf_text_config = _text_config(
        hidden_size=4096, norm_topk_prob=False, num_hidden_layers=80
    )
    with pytest.raises(ValueError) as e:
        runtime.specialized_decode_config(config)
    msg = str(e.value)
    for why in (
        "TP 4 (TP2 only)",
        "expert parallelism",
        "LoRA",
        "speculative decoding",
        "GDN prefix caching (mamba_cache_mode 'align'",
        "GDN cache dtypes",
        "an unfused shared expert",
        "the V1 model runner",
        "model shapes (4096,",
        "router weights without top-k renormalization",
        "80 decoder layers (at most 64)",
        "without --monokernel, vLLM's decode runs",
    ):
        assert why in msg, (why, msg)


@pytest.mark.parametrize(
    "over, why",
    [
        (dict(pipeline_parallel_size=2), "pipeline parallelism"),
        (dict(data_parallel_size=2), "data parallelism"),
        (dict(enable_eplb=True), "EPLB"),
        (dict(decode_context_parallel_size=2), "decode context parallelism"),
        (dict(prefill_context_parallel_size=2), "prefill context parallelism"),
        (dict(use_ubatching=True), "DBO"),
        (dict(enable_hc_sp=True), "sequence parallelism"),
        (dict(use_sequence_parallel_moe=True), "sequence parallelism"),
    ],
)
def test_parallel_refusals(supported_platform, over, why):
    got = runtime.unsupported(_vllm_config(**over))
    assert len(got) == 1 and why in got[0], got


def test_feature_refusals(supported_platform):
    def reasons(edit):
        config = _vllm_config()
        edit(config)
        return runtime.unsupported(config)

    assert reasons(lambda c: None) == []
    cases = {
        "KV transfer": lambda c: setattr(c, "kv_transfer_config", object()),
        "routed-expert capture": lambda c: setattr(
            c.aux_output_config, "enable_return_routed_experts", True
        ),
        "sleep mode": lambda c: setattr(c.model_config, "enable_sleep_mode", True),
        "CPU weight offload": lambda c: setattr(
            c.offload_config.uva, "cpu_offload_gb", 1
        ),
        "weight updates": lambda c: setattr(c, "weight_transfer_config", object()),
        "torch.compile": lambda c: setattr(
            c.compilation_config, "mode", CompilationMode.STOCK_TORCH_COMPILE
        ),
    }
    for why, edit in cases.items():
        got = reasons(edit)
        assert len(got) == 1 and why in got[0], (why, got)


def test_piecewise_graphs_warn(supported_platform):
    config = _vllm_config()
    assert not runtime.piecewise_graphs_only(config)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    assert runtime.piecewise_graphs_only(config)
    assert runtime.specialized_decode_config(config) is not None
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    assert not runtime.piecewise_graphs_only(config)


def _gdn_metadata(decodes=4, prefills=0, spec=0, indices=None):
    return SimpleNamespace(
        num_prefills=prefills,
        num_decodes=decodes,
        num_spec_decodes=spec,
        spec_sequence_masks=None if spec == 0 else torch.ones(1, dtype=torch.bool),
        non_spec_state_indices_tensor=(
            torch.arange(decodes, dtype=torch.int32) if indices is None else indices
        ),
    )


@pytest.mark.parametrize(
    "md, width, why",
    [
        ({"gdn": _gdn_metadata(4)}, 4, None),
        # a FULL graph's pad rows: the indices cover the padded width
        ({"gdn": _gdn_metadata(3, indices=torch.zeros(8, dtype=torch.int32))}, 8, None),
        (None, 4, "no metadata"),
        ({}, 4, "not pure decode"),
        ({"gdn": _gdn_metadata(3, prefills=1)}, 4, "not pure decode"),
        ({"gdn": _gdn_metadata(2, spec=2)}, 4, "spec-verify"),
        (
            {"gdn": _gdn_metadata(0, indices=torch.zeros(4, dtype=torch.int32))},
            4,
            "no decode",
        ),
        (
            {"gdn": _gdn_metadata(4, indices=torch.arange(4))},
            4,
            "no int32 state indices",
        ),
        ({"gdn": _gdn_metadata(4)}, 8, "state indices do not cover the rows"),
        (
            {"gdn": _gdn_metadata(4, indices=torch.zeros(8, dtype=torch.int32)[::2])},
            4,
            "state indices do not cover the rows",
        ),
    ],
)
def test_step_reason(md, width, why):
    assert step_reason(md, "gdn", width) == why
