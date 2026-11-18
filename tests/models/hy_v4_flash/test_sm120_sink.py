# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash on compute capability 12: sink forwarding and NVFP4 expert names."""

import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.model import _EXPERT_WEIGHT
from vllm.models.hy_v4_flash.nvidia.sm120 import (
    FlashSM120Impl,
    FlashSM120SparseBackend,
    FlashSM120SWABackend,
)
from vllm.utils import flashinfer as fi_utils


def test_sm120_flash_backends_apply_sinks_under_the_shared_name() -> None:
    # The shared name keeps the fp8_ds_mla KV layout selection.
    assert FlashSM120SparseBackend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert FlashSM120SparseBackend.supports_sink()
    assert FlashSM120SparseBackend.get_impl_cls() is FlashSM120Impl
    assert FlashSM120SWABackend.supports_sink()
    assert FlashSM120SWABackend.supports_sliding_window()


@pytest.mark.parametrize("has_sink", [True, False])
def test_sm120_flash_impl_forwards_the_sink_to_the_kernel(
    monkeypatch, has_sink: bool
) -> None:
    seen = {}

    def fake_decode(**kwargs):
        seen.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        fi_utils, "flashinfer_trtllm_batch_decode_with_kv_cache_mla", fake_decode
    )
    heads, topk = 16, 512
    impl = object.__new__(FlashSM120Impl)
    impl.num_heads, impl.kv_lora_rank = heads, 512
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = 512, 64
    impl.scale, impl.kv_scale_format = 256**-0.5, "pow2_fp32"
    impl._workspace_buffer = torch.empty(1)
    impl.sinks = torch.arange(heads, dtype=torch.float32) if has_sink else None

    q = torch.zeros(3, heads, 576, dtype=torch.bfloat16)
    kv_cache = torch.zeros(4, 64, 656, dtype=torch.uint8)
    indices = torch.zeros(3, topk, dtype=torch.int32)
    out = impl._run_mqa_kernel(q, kv_cache, indices)

    assert out.shape == (3, heads, 512)
    assert seen["sparse_mla_top_k"] == topk
    if has_sink:
        assert len(seen["sinks"]) == 1 and seen["sinks"][0] is impl.sinks
    else:
        assert seen["sinks"] is None


@pytest.mark.parametrize(
    ("name", "expert", "projection", "suffix"),
    [
        ("layers.1.mlp.experts.7.gate_proj.weight", "7", "gate_proj", ".weight"),
        (
            "layers.1.mlp.experts.7.up_proj.weight_scale_2",
            "7",
            "up_proj",
            ".weight_scale_2",
        ),
        (
            "layers.1.mlp.experts.7.down_proj.input_scale",
            "7",
            "down_proj",
            ".input_scale",
        ),
        (
            "layers.1.mlp.experts.gate_up_proj_scale_inv",
            None,
            "gate_up_proj",
            "_scale_inv",
        ),
    ],
)
def test_expert_weight_names_cover_modelopt_nvfp4(
    name: str, expert: str | None, projection: str, suffix: str
) -> None:
    match = _EXPERT_WEIGHT.fullmatch(name)
    assert match is not None
    assert match.groups()[1:] == (expert, projection, suffix)
