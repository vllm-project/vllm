# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_K31_BEAT_EAGERPREP: the fused one-launch KDA spec-decode staging and
the one-launch num_accepted gather must produce exactly what the reference
(multi-launch) paths produce."""

import dataclasses

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDAMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

DEVICE = torch.device("cuda")
BLOCK_SIZE = 16
NUM_SPEC = 7


def _builder(mode: str, adaptive: bool, max_num_seqs: int, num_spec: int = NUM_SPEC):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=num_spec,
        enable_adaptive_verification=adaptive,
    )
    vllm_config.cache_config.mamba_cache_mode = mode
    # Large enough that every verify case below is FULL-graph staged.
    vllm_config.compilation_config.max_cudagraph_capture_size = 64
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=num_spec,
        mamba_cache_mode=mode,
    )
    b = KimiK3ROCmKDAMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    # Garbage in the staging buffers, identical across the two builders.
    g = torch.Generator(device="cpu").manual_seed(0)
    for name in (
        "spec_state_indices_tensor",
        "spec_token_indx",
        "spec_query_start_loc",
        "num_accepted_tokens",
        "non_spec_token_indx",
    ):
        buf = getattr(b, name)
        buf.copy_(torch.randint(-50, 50, buf.shape, generator=g, dtype=buf.dtype))
    b.spec_sequence_masks.copy_(
        torch.randint(0, 2, b.spec_sequence_masks.shape, generator=g).bool()
    )
    return b


def _batch(query_lens, seq_lens, num_drafts, width_pad, noncontig, num_spec):
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    m = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    max_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE + width_pad
    bt = torch.randint(1, 10000, (len(seq_lens), max_blocks * 2), dtype=torch.int32)
    bt = bt.to(DEVICE)
    bt = bt[:, ::2] if noncontig else bt[:, :max_blocks].contiguous()
    m = m.replace(block_table_tensor=bt)
    accepted = torch.randint(1, num_spec + 2, (len(seq_lens),), dtype=torch.int32)
    accepted = accepted.to(DEVICE)
    draft_cpu = torch.tensor(num_drafts, dtype=torch.int32)
    return m, accepted, draft_cpu


def _eq(x, y, name):
    if isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor), name
        assert x.shape == y.shape and x.dtype == y.dtype, name
        assert x.device == y.device, name
        assert torch.equal(x.cpu(), y.cpu()), name
    elif isinstance(x, dict):
        assert isinstance(y, dict) and x.keys() == y.keys(), name
        for k in x:
            _eq(x[k], y[k], f"{name}[{k}]")
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), name
        for i, (a, b) in enumerate(zip(x, y)):
            _eq(a, b, f"{name}[{i}]")
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        _cmp(x, y)
    else:
        assert x == y, (name, x, y)


def _cmp(a, b):
    assert type(a) is type(b)
    for f in dataclasses.fields(a):
        _eq(getattr(a, f.name), getattr(b, f.name), f.name)


# F = num_spec + 1 (a full verify row), D = num_spec.
CASES = [
    # (query_lens, seq_lens, num_decode_draft_tokens, fused path expected)
    (["F"], [360_017], ["D"], True),  # c1 verify
    (["F"], [33], ["D"], True),
    (["F"], ["F"], ["D"], True),  # seq_len == query_len (capture-like)
    (["F", "F", "F"], [100, 2000, 47], ["D", "D", "D"], True),
    (["F", 0, 0], [100, 0, 0], ["D", -1, -1], True),  # trailing padding rows
    ([2, "F"], [100, 64], [1, "D"], True),  # ragged verify
    ([1, "F"], [100, 64], [0, "D"], True),  # spec row without drafts
    (["F", 3], [100, 64], ["D", -1], False),  # verify + prefill -> reference
    ([1, "F"], [100, 64], [-1, "D"], False),  # decode + verify -> reference
    ([0, "F"], [0, 64], [-1, "D"], False),  # padding not trailing -> reference
    # more tokens than decode_cudagraph_max_bs -> not staged -> reference
    (["F"] * 5, [100] * 5, ["D"] * 5, False),
]


def _resolve(vals, num_spec):
    sub = {"F": num_spec + 1, "D": num_spec}
    return [sub.get(v, v) if isinstance(v, str) else v for v in vals]


@pytest.mark.parametrize("num_spec", [3, 6, 7])
@pytest.mark.parametrize("mode", ["align", "none"])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("noncontig", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_fused_spec_staging_matches_reference(
    monkeypatch, num_spec, mode, adaptive, noncontig, case
):
    query_lens, seq_lens, drafts, expect_fused = case
    query_lens = _resolve(query_lens, num_spec)
    seq_lens = _resolve(seq_lens, num_spec)
    drafts = _resolve(drafts, num_spec)
    width_pad = num_spec + 2
    m, accepted, draft_cpu = _batch(
        query_lens, seq_lens, drafts, width_pad, noncontig, num_spec
    )
    # decode_cudagraph_max_bs = max_num_seqs * (num_spec + 1)
    max_num_seqs = 4
    ref_b = _builder(mode, adaptive, max_num_seqs, num_spec)
    new_b = _builder(mode, adaptive, max_num_seqs, num_spec)

    monkeypatch.setenv("VLLM_K31_BEAT_EAGERPREP", "0")
    ref = ref_b.build(0, m, accepted.clone(), draft_cpu.clone())
    torch.accelerator.synchronize()

    # Record what build() itself took, so a broken gate or fallback that
    # silently runs the reference path fails here.
    took = []
    fused_fn = new_b._build_spec_decode_fused

    def spy(*args, **kwargs):
        out = fused_fn(*args, **kwargs)
        took.append(out is not None)
        return out

    monkeypatch.setattr(new_b, "_build_spec_decode_fused", spy)
    monkeypatch.setenv("VLLM_K31_BEAT_EAGERPREP", "1")
    new = new_b.build(0, m, accepted.clone(), draft_cpu.clone())
    torch.accelerator.synchronize()
    assert took == [expect_fused]
    _cmp(ref, new)
    # The FULL-graph staging buffers (whose addresses graphs captured) must
    # hold identical content past the returned views too.
    for name in (
        "spec_state_indices_tensor",
        "spec_sequence_masks",
        "spec_query_start_loc",
        "num_accepted_tokens",
    ):
        assert torch.equal(getattr(ref_b, name), getattr(new_b, name)), name
    n = ref.spec_token_indx.numel() if ref.spec_token_indx is not None else 0
    assert torch.equal(ref_b.spec_token_indx[:n], new_b.spec_token_indx[:n])
    if expect_fused:
        assert new.spec_state_indices_tensor.data_ptr() == (
            new_b.spec_state_indices_tensor.data_ptr()
        )


@pytest.mark.parametrize("num_real,num_padded", [(1, 1), (1, 4), (3, 3), (5, 300)])
def test_gather_num_accepted(num_real, num_padded):
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import gather_num_accepted

    src = torch.randint(1, 9, (512,), dtype=torch.int32, device=DEVICE)
    idx = torch.randperm(512, device=DEVICE)[:num_real].to(torch.int64)
    if num_real > 2:
        idx[1] = -1  # torch indexing wraps negatives
    ref = src.new_ones(num_padded)
    ref[:num_real] = src[idx]
    out = gather_num_accepted(src, idx, num_padded)
    assert torch.equal(ref, out)


@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ([8], [360_017]),
        ([8], [8]),
        ([8], [100]),
        ([1, 1, 1], [5000, 3, 2048]),
        ([8, 8, 0], [3000, 9, 0]),
        ([5, 8, 3], [2050, 2047, 2049]),
        ([0, 8], [0, 64]),
        ([64] * 4, [10_000, 70, 2040, 64]),
    ],
)
@pytest.mark.parametrize("indptr_len", [257, 16385, 7])
def test_sparse_indptr_fused_matches_reference(query_lens, seq_lens, indptr_len):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        generate_sparse_seqlen_triton,
        sparse_indptr_fused,
    )

    topk = 2048
    qsl = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    num_tokens = int(qsl[-1])
    if indptr_len < num_tokens + 1:
        indptr_len = num_tokens + 1
    qsl = qsl.to(DEVICE)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    garbage = torch.randint(-99, 99, (indptr_len,), dtype=torch.int32, device=DEVICE)
    ref = garbage.clone()
    sp = generate_sparse_seqlen_triton(sl, qsl, topk, num_tokens, max(query_lens))
    torch.cumsum(sp, dim=0, out=ref[1 : num_tokens + 1])
    tail = ref[num_tokens + 1 :]
    tail.copy_(ref[num_tokens].expand_as(tail))
    new = garbage.clone()
    assert sparse_indptr_fused(sl, qsl, new, topk, num_tokens)
    torch.accelerator.synchronize()
    assert torch.equal(ref, new)


def test_sparse_indptr_fused_falls_back_past_one_block():
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        _SPARSE_INDPTR_BLOCK_T,
        sparse_indptr_fused,
    )

    num_tokens = _SPARSE_INDPTR_BLOCK_T + 1
    qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=DEVICE)
    sl = torch.tensor([num_tokens + 100], dtype=torch.int32, device=DEVICE)
    indptr = torch.full((num_tokens + 8,), -5, dtype=torch.int32, device=DEVICE)
    assert not sparse_indptr_fused(sl, qsl, indptr, 2048, num_tokens)
    assert bool((indptr == -5).all())  # nothing launched
