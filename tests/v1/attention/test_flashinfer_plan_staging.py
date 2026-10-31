# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec

if not current_platform.is_cuda():
    pytest.skip("FlashInfer requires CUDA.", allow_module_level=True)

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    FlashInferMetadataBuilder,
    fast_plan_decode,
)

MODEL = "Qwen/Qwen2.5-0.5B"
PAGE_SIZE = 16
NUM_PAGES = 4096


def _mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
    head_size = vllm_config.model_config.get_head_size()
    return {
        name: PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=1.0 / (head_size**0.5),
        )
        for name in layer_names
    }


def _decode_inputs(seq_lens, seed):
    generator = torch.Generator().manual_seed(seed)
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor([cdiv(n, PAGE_SIZE) for n in seq_lens]).cumsum(0)
    indices = torch.randperm(NUM_PAGES, generator=generator)[: int(indptr[-1])]
    last_page_len = torch.tensor(
        [(n - 1) % PAGE_SIZE + 1 for n in seq_lens], dtype=torch.int32
    )
    return dict(
        indptr_cpu=indptr.pin_memory(),
        indices=indices.to(dtype=torch.int32, device="cuda"),
        last_page_len_cpu=last_page_len.pin_memory(),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32).pin_memory(),
    )


def _plan_decode(builder, wrapper, inputs):
    fast_plan_decode(
        wrapper,
        **inputs,
        num_qo_heads=builder.num_qo_heads,
        num_kv_heads=builder.num_kv_heads,
        head_dim=builder.head_dim,
        page_size=builder.page_size,
        sm_scale=builder.sm_scale,
        q_data_type=builder.q_data_type_decode,
        kv_data_type=builder.kv_cache_dtype,
        o_data_type=builder.model_config.dtype,
        fixed_split_size=builder.decode_fixed_split_size,
        disable_split_kv=builder.disable_split_kv,
    )


@pytest.mark.parametrize("use_cudagraph", [False, True])
@torch.inference_mode()
def test_replan_keeps_queued_plan(use_cudagraph, monkeypatch):
    """A plan whose copy to the GPU is still queued keeps its own metadata when
    the wrapper is planned again, as the drafter does every draft step."""
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
        _mock_get_per_layer_parameters,
    )
    config = create_vllm_config(model_name=MODEL, max_model_len=8192)
    spec = FullAttentionSpec(
        block_size=PAGE_SIZE,
        num_kv_heads=config.model_config.get_num_kv_heads(config.parallel_config),
        head_size=config.model_config.get_head_size(),
        dtype=torch.bfloat16,
    )
    lens_a = [37, 5000, 129, 2048, 17, 700, 3333, 64]
    inputs_a = _decode_inputs(lens_a, seed=0)
    inputs_b = _decode_inputs(lens_a[::-1], seed=1)
    with set_current_vllm_config(config):
        builder = FlashInferMetadataBuilder(
            spec, ["layer.0"], config, torch.device("cuda")
        )
        wrapper = builder._get_decode_wrapper(len(lens_a), use_cudagraph)
        workspace = wrapper._int_workspace_buffer

        _plan_decode(builder, wrapper, inputs_a)
        torch.accelerator.synchronize()
        plan_a = workspace.clone()
        _plan_decode(builder, wrapper, inputs_b)
        torch.accelerator.synchronize()
        assert not torch.equal(workspace, plan_a)

        torch.cuda._sleep(100_000_000)  # hold plan A's copy in the queue
        _plan_decode(builder, wrapper, inputs_a)
        plan_a_copied = torch.cuda.Event()
        plan_a_copied.record()
        seen_by_a = workspace.clone()
        _plan_decode(builder, wrapper, inputs_b)
        replanned_while_queued = not plan_a_copied.query()
        torch.accelerator.synchronize()

    assert replanned_while_queued
    assert torch.equal(seen_by_a, plan_a)
