# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent KV-cache writes and reads of ROCM_AITER_MLA_SPARSE.

The layer writes ``[kv_c ; k_pe]`` rows through ``do_kv_cache_update``; the
sparse kernels read them back as flat rows at ``block * block_stride_rows +
offset``. Checked per KV layout (including a multi-layer layout whose blocks
are strided by other layers' pages), cache dtype and block size.
"""

import pytest
import torch

from tests.v1.attention.backends.mla_harness import fp8_representable
from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    DEVICE,
    FP8_K_SCALE,
    make_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2, GLM_5_3_FLASH
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.kv_cache_interface import KVCacheLayout

DSV32 = DEEPSEEK_V3_2.shard(8)
LAYOUTS = [KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.BLHNC]


def _read_rows(case, layer, slots: torch.Tensor) -> torch.Tensor:
    rows, stride = flat_kv_row_view(layer.kv_cache, case.block_size)
    flat = (slots // case.block_size) * stride + slots % case.block_size
    out = rows[flat]
    if case.quantized:
        fp8 = current_platform.fp8_dtype()
        out = out.view(fp8).float() * FP8_K_SCALE
    return out.float()


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda x: x.name)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("shape", [DSV32, GLM_5_3_FLASH.shard(8)], ids=lambda s: s.id)
def test_write_read_round_trip(env, layout, kv_cache_dtype, block_size, shape):
    case = make_case(
        shape,
        [256],
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        layout=layout,
        num_layers=2,
    )
    num_slots = case.num_blocks * block_size
    slots = torch.randperm(num_slots, device=DEVICE)[:200]
    for layer in case.layers.values():
        torch.testing.assert_close(
            _read_rows(case, layer, slots), case.kv_rows[slots].float(), rtol=0, atol=0
        )
        rows, stride = flat_kv_row_view(layer.kv_cache, block_size)
        assert stride >= block_size


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_layer_write_skips_padding_slots(env, kv_cache_dtype):
    case = make_case(DSV32, [256], kv_cache_dtype=kv_cache_dtype)
    layer = case.layer
    lora = case.shape.kv_lora_rank
    slots = torch.tensor([3, -1, 5, -1], device=DEVICE)
    new = torch.randn(4, case.shape.head_size, dtype=case.dtype, device=DEVICE)
    if case.quantized:
        new = fp8_representable(new, FP8_K_SCALE)
    before = _read_rows(case, layer, torch.arange(8, device=DEVICE))
    layer.impl.do_kv_cache_update(
        new[:, :lora],
        new[:, lora:].unsqueeze(1),
        layer.kv_cache,
        slots,
        layer.kv_cache_dtype,
        layer._k_scale,
    )
    after = _read_rows(case, layer, torch.arange(8, device=DEVICE))
    expected = before.clone()
    expected[3], expected[5] = new[0].float(), new[2].float()
    torch.testing.assert_close(after, expected, rtol=0, atol=0)


def test_fp8_write_applies_the_k_scale(env):
    """A missing scale would store x instead of x / k_scale."""
    case = make_case(DSV32, [64], kv_cache_dtype="fp8")
    rows, _ = flat_kv_row_view(case.layer.kv_cache, case.block_size)
    stored = rows[:64].view(current_platform.fp8_dtype()).float()
    expected = case.kv_rows[:64].float() / FP8_K_SCALE
    torch.testing.assert_close(stored, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_layouts_give_identical_attention(env, kv_cache_dtype):
    query_lens, seq_lens = [1, 1, 64], [300, 700, 500]
    outputs = []
    for layout in LAYOUTS:
        case = make_case(
            DSV32,
            seq_lens,
            kv_cache_dtype=kv_cache_dtype,
            layout=layout,
            num_layers=2,
            force_mqa=True,
        )
        batch = case.batch(query_lens, seq_lens)
        query = case.query(batch.num_tokens)
        indices = case.indices(batch)
        out = case.run(batch, query, indices)
        for name, layer in case.layers.items():
            case.check(batch, query, indices, out[name], layer=layer)
        outputs.append(out[case.names[0]])
    for other in outputs[1:]:
        torch.testing.assert_close(other, outputs[0], rtol=0, atol=0)
