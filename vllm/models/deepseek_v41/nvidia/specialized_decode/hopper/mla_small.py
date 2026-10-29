# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The small-batch sparse-MLA decode of a Hopper executor step, up to
``MAX_ROWS`` rows of uniform verify steps.

FlashMLA pads each row's 16 local heads to a 64-row tile. These kernels
(``mla_cute``) pack up to four rows of one request into a tile instead, on the
stock caches, SWA indices and the indexer's local top-k indices:

- At the start of the step, on a side stream, ``UnionBuild`` builds the union
  of each row group's SWA slots once per SWA KV-cache group.
- The first layer of an index source (``sparse_index_key``) runs ``CuPrep3``:
  the union of the group's compressed top-k slots mapped through the compressed
  block table (``compute_global_topk_indices_and_lens`` fused), dequantized
  into tile images. The source layer is the first reader of its top-k, so this
  runs on the forward stream.
- Each layer runs ``SmallDecode`` and ``Combine`` into heads 0..15 of the stock
  padded output.
"""

import math
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.forward_context import get_forward_context

from ...flashmla import DeepseekV4FlashMLAAttention
from ..attention.sparse_mla import sparse_index_key

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# Above it FlashMLA's padded tiles are full enough to win.
MAX_ROWS = 12
_HEADS = 16
_ROWS_PER_GROUP = 64 // _HEADS
_SWA_WIDTH, _TOP_K = 128, 512


def _log2(n: int) -> int:
    assert n > 0 and n & (n - 1) == 0, n
    return n.bit_length() - 1


@cache
def _compiled(kind: str, compressed: bool):
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from . import mla_cute

    i, p = Int32(1), Int64(0)
    if kind == "union":
        kernel: Any = mla_cute.UnionBuild(_ROWS_PER_GROUP)
        args = [p, i, i, p, p, i, i, p, p, p, p, i, i, i, i]
    elif kind == "prep":
        kernel = mla_cute.CuPrep3(_HEADS)
        args = [p, i, i, p, p, p, p, i, i, p, p, i, p, p, p, i, i, i]
    elif kind == "decode":
        kernel = mla_cute.SmallDecode(_HEADS, compressed)
        args = [p, i, i, p, i, i, p, p, i, p, p, p, i, p, p, p, p, Float32(1.0)]
        args += [p, p, i, i, i]
    else:
        kernel = mla_cute.Combine(_HEADS, compressed)
        args = [p, p, p, p, i, i, i, i, i]
    return cute.compile(kernel, *args, make_fake_stream())


def _launch(kind: str, compressed: bool, *args: Any) -> None:
    """``kind``'s kernel on the current stream; tensors pass their address,
    floats and ``("i64", n)`` pass as 64-bit, other ints as Int32."""
    from cutlass import Float32, Int32, Int64

    from ...ops.cute_dsl.primitives import current_cu_stream

    values: list = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            values.append(Int64(arg.data_ptr()))
        elif isinstance(arg, tuple):
            values.append(Int64(arg[1]))
        elif isinstance(arg, float):
            values.append(Float32(arg))
        else:
            values.append(Int32(arg))
    _compiled(kind, compressed)(*values, current_cu_stream())


class SmallMlaDecode:
    """The unions, compressed stage and partials of a step on the small path,
    in buffers allocated once and sized for ``MAX_ROWS`` rows."""

    def __init__(
        self, attns: list[DeepseekV4FlashMLAAttention], device: torch.device
    ) -> None:
        from .mla_cute import IMG, NBC, NSWB

        self.keys = {attn: sparse_index_key(attn) for attn in attns}
        self.supported = all(
            attn.n_local_heads == _HEADS
            and attn.padded_heads == 64
            and (
                attn.compress_ratio == 0
                or (
                    attn.topk_indices_buffer is not None
                    and attn.topk_indices_buffer.shape[-1] <= _TOP_K
                )
            )
            for attn in attns
        )
        # A row group per row at most.
        groups = MAX_ROWS
        width = NSWB * 64
        # Per SWA cache layer; a step builds one union per SWA KV-cache group
        # (metadata) into its first layer's buffers.
        self._buffers = {
            prefix: (
                torch.full((groups, width), -1, dtype=torch.int32, device=device),
                torch.zeros(groups, width, dtype=torch.uint8, device=device),
                torch.zeros(groups, dtype=torch.int32, device=device),
            )
            for prefix in sorted({attn.swa_cache_layer.prefix for attn in attns})
        }
        self.stage = (
            torch.zeros(groups, NBC, IMG, dtype=torch.float16, device=device),
            torch.zeros(groups, NBC * 64, dtype=torch.uint8, device=device),
            torch.zeros(groups, dtype=torch.int32, device=device),
        )
        slots = NSWB + NBC
        self.work_o = torch.empty(
            groups * slots * 64 * 512, dtype=torch.bfloat16, device=device
        )
        self.work_ml = torch.empty(
            groups * slots * 64 * 2, dtype=torch.float32, device=device
        )
        self.unions: dict[str, tuple[torch.Tensor, ...]] = {}
        self.empty = torch.zeros(16, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device)
        self.union_done = torch.cuda.Event()
        # The step's rows per request, or 0 when it runs FlashMLA.
        self.query_len = 0
        self.groups = 0
        self._stage_key: str | None = None
        self._forked = False

    def _serves(self, metadata: "DeepseekSparseSWAMetadata") -> int:
        """The step's rows per request if it takes the small path, else 0."""
        rows, requests = metadata.num_decode_tokens, metadata.num_decodes
        query_len = metadata.max_decode_query_len
        indices = metadata.decode_swa_indices
        if (
            not self.supported
            or not 0 < rows <= MAX_ROWS
            or rows != requests * query_len
            or indices is None
            or indices.shape[-1] > _SWA_WIDTH
        ):
            return 0
        return query_len

    def start(self) -> None:
        """Choose the step's path; on the small path, fork the SWA unions."""
        self._stage_key = None
        by_layer = get_forward_context().attn_metadata
        assert isinstance(by_layer, dict)
        owners: dict[int, str] = {}
        for prefix in self._buffers:
            owners.setdefault(id(by_layer[prefix]), prefix)
        self.unions = {
            prefix: self._buffers[owners[id(by_layer[prefix])]]
            for prefix in self._buffers
        }
        metadata = {
            prefix: cast("DeepseekSparseSWAMetadata", by_layer[prefix])
            for prefix in owners.values()
        }
        first = next(iter(metadata.values()))
        self.query_len = self._serves(first)
        if not self.query_len:
            return
        per_request = -(-self.query_len // _ROWS_PER_GROUP)
        self.groups = first.num_decodes * per_request
        self.stream.wait_stream(torch.cuda.current_stream())
        self._forked = True
        with torch.cuda.stream(self.stream):
            for prefix, group in metadata.items():
                kidx, kmask, klen = self._buffers[prefix]
                indices, lens = group.decode_swa_indices, group.decode_swa_lens
                assert indices is not None and lens is not None
                indices = indices.view(group.num_decode_tokens, -1)
                swa = (indices, indices.stride(0), indices.shape[-1], lens)
                _launch(
                    "union",
                    False,
                    *swa,
                    *swa,
                    kidx,
                    kmask,
                    klen,
                    kidx.shape[1],
                    self.query_len,
                    per_request,
                    self.groups,
                )
            self.union_done.record()

    def join(self) -> None:
        """Wait for the unions if no layer has yet."""
        if self._forked:
            torch.cuda.current_stream().wait_event(self.union_done)
            self._forked = False

    def run(
        self,
        attn: DeepseekV4FlashMLAAttention,
        q: torch.Tensor,
        out: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: "DeepseekV4FlashMLAMetadata | None",
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Write ``attn``'s sparse MLA of ``q`` (padded heads) into heads
        0..15 of ``out``; or without ``out``, return it inverse-rotated at
        ``positions`` as the ``wo_a`` input [rows, groups, heads per group *
        512] (``attn_chain``)."""
        self.join()
        kidx, kmask, klen = self.unions[attn.swa_cache_layer.prefix]
        compressed = attn.compress_ratio != 0
        img = mbits = count = self.empty
        if compressed:
            assert attn_metadata is not None
            img, mbits, count = self._prepare(attn, swa_metadata, attn_metadata)
        swa_cache = attn.swa_cache_layer.kv_cache
        per_request = -(-self.query_len // _ROWS_PER_GROUP)
        shape = (self.query_len, per_request, self.groups)
        # The decode writes only its partials; the combine writes the output.
        _launch(
            "decode",
            compressed,
            q,
            q.stride(0),
            q.stride(1),
            self.empty,
            0,
            0,
            swa_cache,
            ("i64", swa_cache.stride(0)),
            _log2(swa_cache.shape[1]),
            kidx,
            kmask,
            klen,
            kidx.shape[1],
            img,
            mbits,
            count,
            attn.attn_sink,
            attn.scale * math.log2(math.e),
            self.work_o,
            self.work_ml,
            *shape,
        )
        if out is not None:
            _launch(
                "combine",
                compressed,
                self.work_o,
                self.work_ml,
                attn.attn_sink,
                out,
                out.stride(0),
                out.stride(1),
                *shape,
            )
            return None
        from . import attn_chain

        assert positions is not None
        rows = q.shape[0]
        groups = attn.n_local_groups
        rotated = torch.empty(
            rows,
            groups,
            attn.n_local_heads // groups * 512,
            dtype=torch.bfloat16,
            device=q.device,
        )
        attn_chain.launch(
            "combine",
            40 if compressed else 8,
            self.work_o,
            self.work_ml,
            attn.attn_sink,
            rotated,
            rotated.stride(0),
            rotated.stride(1),
            positions,
            attn.rotary_emb.cos_sin_cache,
            *shape,
        )
        return rotated

    def _prepare(
        self,
        attn: DeepseekV4FlashMLAAttention,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: "DeepseekV4FlashMLAMetadata",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The compressed stage of ``attn``'s index source, prepared by its
        first layer: an index source's layers are consecutive."""
        key = self.keys[attn]
        if key == self._stage_key:
            return self.stage
        self._stage_key = key
        topk = attn.topk_indices_buffer
        tok2req = swa_metadata.token_to_req_indices
        valid = swa_metadata.is_valid_token
        assert topk is not None and tok2req is not None and valid is not None
        table = attn_metadata.block_table
        cache = attn._compressed_kv_cache()
        img, mbits, count = self.stage
        per_request = -(-self.query_len // _ROWS_PER_GROUP)
        _launch(
            "prep",
            True,
            topk,
            topk.stride(0),
            topk.shape[-1],
            # The top-k lengths, unused when mapping local indices.
            tok2req,
            tok2req,
            valid,
            table,
            table.stride(0),
            _log2(attn_metadata.block_size // attn.compress_ratio),
            cache,
            ("i64", cache.stride(0)),
            _log2(cache.shape[1]),
            img,
            mbits,
            count,
            self.query_len,
            per_request,
            self.groups,
        )
        return self.stage

    def warm_up(self) -> None:
        """Compile the kernels before CUDA graph capture."""
        if self.supported:
            for kind, compressed in (
                ("union", False),
                ("prep", True),
                ("decode", False),
                ("decode", True),
                ("combine", False),
                ("combine", True),
            ):
                _compiled(kind, compressed)
