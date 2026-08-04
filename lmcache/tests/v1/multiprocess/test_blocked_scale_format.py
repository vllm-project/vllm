# SPDX-License-Identifier: Apache-2.0
"""NL_X_NB_BSV_BSS: blocked-scale DSA indexer cache transfers.

vLLM lays an indexer-cache page out blocked — per 64-token block, all tokens'
128-byte fp8 values first, then all tokens' 4-byte fp32 scales. LMCache's
chunk rows are canonical token-major ``[vals | scale]`` (132 B). These tests
build a synthetic blocked page, gather it token-granular (D2H), scatter to a
DIFFERENTLY-ALIGNED destination (H2D), and require value-exactness — the
mod-block_size misalignment that silently garbled GLM CacheBlend reuse.
"""

# Third Party
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

# First Party
import lmcache.c_ops as lmc_ops  # noqa: E402

if not hasattr(lmc_ops.EngineKVFormat, "NL_X_NB_BSV_BSS"):
    pytest.skip("c_ops build lacks NL_X_NB_BSV_BSS", allow_module_level=True)

_BS = 64  # tokens per block
_HD = 128  # fp8 value bytes per token
_ROW = _HD + 4  # + fp32 scale
_NB = 8
_NL = 3


def _blocked_page_write(cache, layer, slot, row_bytes):
    """Write one token's 132-byte row into the BLOCKED page layout (the
    ground truth from vllm's indexer_k_quant_and_cache)."""
    b, off = slot // _BS, slot % _BS
    flat = cache[layer].view(-1)
    base = b * _BS * _ROW
    flat[base + off * _HD : base + off * _HD + _HD] = row_bytes[:_HD]
    sbase = base + _BS * _HD
    flat[sbase + off * 4 : sbase + off * 4 + 4] = row_bytes[_HD:]


def _blocked_page_read(cache, layer, slot):
    b, off = slot // _BS, slot % _BS
    flat = cache[layer].view(-1)
    base = b * _BS * _ROW
    vals = flat[base + off * _HD : base + off * _HD + _HD]
    sbase = base + _BS * _HD
    scale = flat[sbase + off * 4 : sbase + off * 4 + 4]
    return torch.cat([vals, scale])


def _make_paged():
    """Per-layer [NB, BS, 132] uint8 tensors + a pointer array."""
    caches = [
        torch.zeros(_NB, _BS, _ROW, dtype=torch.uint8, device="cuda")
        for _ in range(_NL)
    ]
    ptrs = torch.tensor(
        [c.data_ptr() for c in caches], dtype=torch.int64, device="cuda"
    )
    return caches, ptrs


def _write_all(caches, slots, rows):
    for layer in range(_NL):
        for i, slot in enumerate(slots.tolist()):
            _blocked_page_write(caches, layer, slot, rows[layer, i])


def _read_all(caches, slots):
    out = []
    for layer in range(_NL):
        out.append(
            torch.stack([_blocked_page_read(caches, layer, s) for s in slots.tolist()])
        )
    return torch.stack(out)


def test_blocked_roundtrip_value_exact_across_alignments():
    torch.manual_seed(0)
    n_tok = 96
    src, src_ptrs = _make_paged()
    rows = torch.randint(0, 255, (_NL, n_tok, _ROW), dtype=torch.uint8, device="cuda")
    src_slots = torch.arange(32, 32 + n_tok, dtype=torch.int64, device="cuda")
    _write_all(src, src_slots, rows)

    # D2H-style gather into a token-major chunk buffer (kv=1, NL, n_tok, 132).
    chunk = torch.zeros(1, _NL, n_tok, _ROW, dtype=torch.uint8, device="cuda")
    lmc_ops.multi_layer_kv_transfer(
        chunk,
        src_ptrs,
        src_slots,
        torch.device("cuda"),
        _NB * _BS,
        lmc_ops.TransferDirection.D2H,
        lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS,
        block_size=_BS,
        head_size=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(chunk[0], rows), "gather must produce token-major rows"

    # H2D scatter to a DIFFERENT intra-block alignment (delta 17 mod 64 != 0).
    dst, dst_ptrs = _make_paged()
    dst_slots = torch.arange(17, 17 + n_tok, dtype=torch.int64, device="cuda")
    lmc_ops.multi_layer_kv_transfer(
        chunk,
        dst_ptrs,
        dst_slots,
        torch.device("cuda"),
        _NB * _BS,
        lmc_ops.TransferDirection.H2D,
        lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS,
        block_size=_BS,
        head_size=0,
    )
    torch.cuda.synchronize()
    got = _read_all(dst, dst_slots)
    assert torch.equal(got, rows), (
        "misaligned scatter must land every token's values AND scale at the "
        "blocked-layout positions of the destination slots"
    )
