# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse indexer's decode top-512 for gfx942 (topk512_gfx942.cu).

``build`` builds the extension with hipcc into torch's extension cache (about
a minute), and later processes load the cached build. The mono decode layers
call it when the model is created, so that the build happens at startup and
not at the first decode step. The TP ranks share one build: torch's extension
loader makes the other ranks wait for the rank that builds it. When the build
fails, every hook here returns False and vLLM runs its own ops.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.dsv41_gfx942 import enabled

logger = init_logger(__name__)

_HERE = os.path.dirname(os.path.realpath(__file__))
_SOURCE = os.path.join(_HERE, "topk512_gfx942.cu")

# A row is split into one block for every MIN_CHUNK live logits, up to
# MAX_BLOCKS blocks. On the indexer logits of one 128k decode step, 8 and
# 16384 took 222 us for the step's 8 calls, against 543 us for vLLM's kernel.
# 16 blocks took 294 us: a block of a masked row then often holds fewer than
# 512 finite logits, and the exact emission of -inf ties is slow.
MAX_BLOCKS = 8
MIN_CHUNK = 16384

_ext = None
_build_failed = False


def _compile():
    from torch.utils.cpp_extension import load

    # The container lists every ROCm target in PYTORCH_ROCM_ARCH. Build for
    # gfx942 only, and restore the list afterwards so other extensions that
    # this process builds are not affected.
    saved = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
    try:
        return load(name="dsv41_topk942", sources=[_SOURCE], extra_cuda_cflags=["-O3"])
    finally:
        if saved is None:
            del os.environ["PYTORCH_ROCM_ARCH"]
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = saved


def build() -> bool:
    """Build or load the extension once, and return whether it is available.
    A failed build, for example without hipcc or without a writable extension
    cache, logs a warning. The hooks then return False and vLLM runs its own
    top-k ops for the rest of the process."""
    global _ext, _build_failed
    if _ext is None and not _build_failed:
        try:
            _ext = _compile()
        except Exception as err:
            _build_failed = True
            logger.warning(
                "DSv4.1 gfx942 top-k: building topk512_gfx942.cu failed, so "
                "vLLM's top-k ops run instead: %s",
                err,
            )
    return _ext is not None


def _ready() -> bool:
    return enabled() and build()


def _load():
    # The hooks call build through _ready first. Direct callers such as the
    # kernel tests get the build here, and an error when it failed.
    if not build():
        raise RuntimeError("topk512_gfx942.cu did not build, see the warning")
    return _ext


def decode_top_k(logits, next_n, seq_lens, indices, topk_tokens) -> bool:
    """The hook in vLLM's ROCm decode indexer (rocm_aiter_mla_sparse.py).

    When ``enabled()`` and the extension is built, a call with topK 512 and
    unit column strides runs here and returns True. Otherwise it returns
    False and vLLM runs torch.ops._C.top_k_per_row_decode as before."""
    if not _ready():
        return False
    taken = topk_tokens == 512 and logits.stride(1) == 1 and indices.stride(1) == 1
    if taken:
        top_k_per_row_decode_512(logits, next_n, seq_lens, indices)
    return taken


def candidate_top_k(
    logits, next_n, seq_lens, candidates, block_size, indices, topk_tokens
) -> bool:
    """The hook in front of the DSpark candidate mask of the decode indexer
    (rocm_aiter_mla_sparse.py), on layers 24 to 36.

    When ``enabled()`` and the extension is built, a call with topK 512
    writes each row's top 512 among its candidate blocks to ``indices`` and
    returns True. vLLM then skips its mask and its top-k. The mask writes -inf
    to about 7 of every 8 logits of a 128k row, and the top-k then reads the
    whole row. Here only the 2048 x 8 candidate logits of a row are read.
    Otherwise this returns False and vLLM masks and runs its top-k as
    before."""
    if not _ready():
        return False
    taken = (
        topk_tokens == 512
        and logits.stride(1) == 1
        and indices.stride(1) == 1
        and candidates.dtype == torch.int32
        and candidates.stride(1) == 1
        and candidates.shape[1] * block_size > 512
    )
    if taken:
        _load().candidate_top_k_512(
            logits, next_n, seq_lens, candidates, block_size, indices
        )
    return taken


def candidate_logits_top_k(
    q_fp8,
    kv_cache,
    weights,
    seq_lens,
    block_table,
    candidates,
    block_size,
    indices,
    topk_tokens,
    dense_is_aiter_gluon,
    requires_padding,
) -> bool:
    """The hook in front of the dense logits of the decode indexer
    (rocm_aiter_mla_sparse.py), on layers 24 to 36.

    When ``enabled()`` and the extension is built, a call that the hook
    takes writes each row's top 512 among its candidate blocks to ``indices``
    and returns True. vLLM then skips its dense logits, its candidate mask and
    its top-k for the layer. The call computes only the 2048 x 8 candidate
    logits of each row (cand_logits.py), bit-identical to the dense kernel's,
    and then runs the same top-512 as candidate_top_k. Otherwise this returns
    False and vLLM runs the dense path as before.

    The hook only takes calls whose dense logits would come from AITER's
    gfx942 Gluon kernel, because cand_logits.py reproduces that kernel's
    arithmetic. dense_is_aiter_gluon says whether they would."""
    if not _ready():
        return False
    rows = q_fp8.shape[0] * q_fp8.shape[1]
    taken = (
        dense_is_aiter_gluon
        and not requires_padding
        and topk_tokens == 512
        and indices.stride(1) == 1
        and q_fp8.dim() == 4
        and q_fp8.shape[2] in (32, 64)
        and q_fp8.shape[3] == 128
        and q_fp8.is_contiguous()
        and kv_cache.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[1] % 16 == 0
        and kv_cache.shape[3] == 128 + 4
        and weights.stride(1) == 1
        and candidates.dtype == torch.int32
        and candidates.stride(1) == 1
        and candidates.shape[0] >= rows
        and candidates.shape[1] * block_size > 512
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
    )
    if not taken:
        return False
    # cand_logits imports AITER's Gluon kernels, so only a call that the
    # hook takes loads it.
    from vllm.model_executor.layers.dsv41_gfx942 import cand_logits

    next_n = q_fp8.shape[1]
    candidates = candidates[:rows]
    row_len = candidates.shape[1] * block_size
    compact_logits = torch.empty(
        rows, row_len, device=q_fp8.device, dtype=torch.float32
    )
    compact_ids = torch.empty(rows, row_len, device=q_fp8.device, dtype=torch.int32)
    cand_logits.candidate_logits(
        q_fp8.view(rows, *q_fp8.shape[2:]),
        kv_cache,
        weights[:rows],
        seq_lens,
        next_n,
        block_table,
        candidates,
        block_size,
        compact_logits,
        compact_ids,
    )
    # selectFromRegisters keeps a row of up to 32768 logits in registers. The
    # compact rows of the 2048 candidate blocks of 8 hold 16384.
    if row_len % 4 == 0 and row_len <= 32768:
        _load().compact_top_k_512_regs(
            compact_logits, compact_ids, seq_lens, next_n, indices
        )
    else:
        _load().compact_top_k_512(
            compact_logits, compact_ids, seq_lens, next_n, indices
        )
    return True


def select_candidates(logits, next_n, seq_lens, block_size, candidates) -> bool:
    """The hook in front of vLLM's select_candidate_blocks in the decode
    indexer (rocm_aiter_mla_sparse.py), on layer 20, which writes the
    candidate blocks of layers 24 to 36.

    When ``enabled()`` and the extension is built, ``candidates`` gets each
    row's block ids from one launch (candidateBlocks in topk512_gfx942.cu) and
    this returns True. vLLM takes three launches and torch.topk, which also
    sorts the picks, and about 100 us for a 128k step's 6 rows. The block ids
    are the same, in no particular order. Otherwise this returns False and
    vLLM runs its version."""
    if not _ready():
        return False
    ends = seq_lens.reshape(-1)
    rows = logits.shape[0]
    taken = (
        logits.stride(1) == 1
        and ends.dtype == torch.int32
        and ends.is_contiguous()
        and candidates.dtype == torch.int32
        and candidates.stride(1) == 1
        and candidates.shape[0] >= rows
        and block_size >= 1
    )
    if taken:
        # vLLM repeats each request's length over its next_n rows when
        # seq_lens holds one length a request (rocm_aiter_mla_sparse.py).
        row_repeat = 1 if ends.numel() == rows else next_n
        _load().candidate_blocks(logits, ends, row_repeat, block_size, candidates)
    return taken


def skip_decode_fill(has_prefill, num_decode_tokens, num_tokens, decode) -> bool:
    """Whether vLLM's decode indexer (rocm_aiter_mla_sparse.py) may skip its
    -1 fill of the step's top-k rows. When ``enabled()`` and the extension is
    built, it may when every row of the step is a decode row and none is
    padded. The decode top-k (decode_top_k) then writes every entry of every
    row, the -1 past a short row's end included. vLLM's own top-k does not,
    so without the extension the fill stays. This saves one launch on each of
    the 8 index layers."""
    return (
        _ready()
        and not has_prefill
        and decode is not None
        and not decode.requires_padding
        and num_decode_tokens == num_tokens
    )


def top_k_per_row_decode_512(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    max_blocks: int = MAX_BLOCKS,
    min_chunk: int = MIN_CHUNK,
    register_select: bool = True,
) -> None:
    """The contract of vLLM's ops.top_k_per_row_decode with topK = 512:
    indices[r, :512] gets the column indices of row r's 512 largest logits, in
    no particular order, and -1 past the row's length when it is shorter.

    With ``register_select`` a row of up to 1M logits whose rows are 16-byte
    aligned runs selectFromRegisters, which keeps a block's logits in
    registers. Otherwise, and for other rows, the row runs the gfx942 build of
    vLLM's radix job. The index sets are the same, except for ties at the
    cut."""
    if (
        register_select
        and logits.stride(0) % 4 == 0
        and logits.data_ptr() % 16 == 0
        and logits.shape[1] <= 64 * 16384
    ):
        _load().top_k_per_row_decode_512_regs(logits, next_n, seq_lens, indices)
        return
    _load().top_k_per_row_decode_512(
        logits, next_n, seq_lens, indices, max_blocks, min_chunk
    )
