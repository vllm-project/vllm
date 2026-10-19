# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.minimax_m3.amd.specialized_decode.runtime.caches import (
    tensor_identity,
)
from vllm.models.minimax_m3.amd.specialized_decode.runtime.decode import Step
from vllm.models.minimax_m3.amd.specialized_decode.runtime.routing import decide_step
from vllm.models.minimax_m3.amd.specialized_decode.runtime.spec import DECODE_CONFIG
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseMetadata,
)


def _metadata(rows: int, *, decode_tokens: int | None = None, prefills: int = 0):
    decode_tokens = rows if decode_tokens is None else decode_tokens
    return MiniMaxM3SparseMetadata(
        seq_lens=torch.ones(rows, dtype=torch.int32),
        max_seq_len=1,
        slot_mapping=torch.arange(rows),
        num_actual_tokens=decode_tokens,
        num_decodes=decode_tokens,
        num_decode_tokens=decode_tokens,
        num_prefills=prefills,
        num_prefill_tokens=prefills,
        decode=MiniMaxM3SparseDecodeMetadata(
            seq_lens=torch.ones(decode_tokens, dtype=torch.int32),
            block_table=torch.zeros(decode_tokens, 1, dtype=torch.int32),
            decode_query_len=1,
        ),
    )


@pytest.mark.parametrize("width", DECODE_CONFIG.widths)
def test_decode_widths_route_uniformly(width: int):
    left = decide_step(DECODE_CONFIG, width, CUDAGraphMode.NONE, _metadata(width))
    right = decide_step(DECODE_CONFIG, width, CUDAGraphMode.NONE, _metadata(width))
    assert left == right
    assert left.use_mono


@pytest.mark.parametrize(
    ("rows", "mode", "metadata"),
    [
        (DECODE_CONFIG.max_rows + 1, CUDAGraphMode.NONE, _metadata(1)),
        (1, CUDAGraphMode.FULL, _metadata(1)),
        (1, CUDAGraphMode.PIECEWISE, _metadata(1)),
        (1, CUDAGraphMode.NONE, None),
        (1, CUDAGraphMode.NONE, _metadata(1, prefills=1)),
        (1, CUDAGraphMode.NONE, _metadata(1, decode_tokens=2)),
    ],
)
def test_unsupported_steps_fall_back(rows, mode, metadata):
    assert not decide_step(DECODE_CONFIG, rows, mode, metadata).use_mono


def test_padding_slots_are_eligible():
    metadata = _metadata(4, decode_tokens=2)
    metadata.slot_mapping[2:] = -1
    assert decide_step(DECODE_CONFIG, 4, CUDAGraphMode.NONE, metadata).use_mono


@pytest.mark.parametrize("query_len", [2, 4])
def test_speculative_verification_falls_back(query_len: int):
    metadata = _metadata(4)
    metadata.decode.decode_query_len = query_len
    assert not decide_step(DECODE_CONFIG, 4, CUDAGraphMode.NONE, metadata).use_mono


def test_cache_identity_detects_replacement():
    first = torch.empty(8)
    replacement = torch.empty(8)
    assert tensor_identity(first, "test") != tensor_identity(replacement, "test")
    assert tensor_identity(first, "test") == tensor_identity(first, "test")


def _config(**overrides):
    values = dict(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_ubatching=False,
    )
    values.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**values),
        cache_config=SimpleNamespace(
            cache_dtype="fp8",
            block_size=128,
            enable_prefix_caching=False,
            kv_offloading_size=None,
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda _default: "fp8"
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 32],
            mode=CompilationMode.NONE,
        ),
        kernel_config=SimpleNamespace(moe_backend="aiter", monokernel=True),
        lora_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector=None),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        speculative_config=None,
        weight_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            enforce_eager=True,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(
                use_index_cache=False,
                num_hidden_layers=60,
                sparse_attention_config={"sparse_attention_freq": [0, 0, 0] + [1] * 57},
            ),
        ),
    )


def test_static_refusals_report_all_reasons():
    config = _config(
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )
    config.cache_config.block_size = 64
    reasons = DECODE_CONFIG.refusal_reasons(config)
    assert any("tensor parallel" in reason for reason in reasons)
    assert any("block size" in reason for reason in reasons)


def test_routed_expert_capture_is_refused():
    config = _config()
    config.aux_output_config.enable_return_routed_experts = True
    reasons = DECODE_CONFIG.refusal_reasons(config)
    assert any("routed-expert capture" in reason for reason in reasons)


def test_step_layer_lookup_is_immutable():
    def kernel(*args):
        return args[1], args[2]

    step = Step(first_layer=3, kernels=(kernel, None))
    assert step.layer(3) is kernel
    assert step.layer(4) is None
    assert step.layer(2) is None
    with pytest.raises(AttributeError):
        step.kernels = ()
