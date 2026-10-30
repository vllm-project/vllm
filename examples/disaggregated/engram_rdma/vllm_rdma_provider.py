# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research bridge from vLLM Engram lookup to the validated NIXL GET shim.

The SGLang research runtime and its native library must be on PYTHONPATH and
configured before importing this module. This is not a public NIXL API.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def verify_checkpoint_samples():
    """Reject mismatched owner weights using published boundary-row samples."""
    from real_checkpoint_owner import inspect_checkpoint

    ready = json.loads(Path(os.environ["ENGRAM_SHARED_READY"]).read_text())
    samples = ready.get("sample_validation")
    if not isinstance(samples, list) or {s.get("layer") for s in samples} != {1, 14}:
        raise ValueError("owner did not publish the required checkpoint samples")
    manifest = inspect_checkpoint(Path(os.environ["ENGRAM_SHARED_MODEL"]))
    if manifest["checkpoint_identity"] != ready["checkpoint_id"]:
        raise ValueError("checkpoint metadata identity differs from owner")
    for sample in samples:
        layer = sample["layer"]
        for kind, width in (("weight", 256), ("scale", 8)):
            spec = next(
                s
                for s in manifest["specs"]
                if s["layer"] == layer and s["kind"] == kind
            )
            required_rows = sorted({0, 1, spec["rows"] // 2, spec["rows"] - 1})
            if sample.get("rows") != required_rows:
                raise ValueError("owner checkpoint sample rows are incomplete")
            shard = manifest["shards"][spec["shard"]]
            digest = hashlib.sha256()
            with open(shard["path"], "rb", buffering=0) as file:
                for row in sample["rows"]:
                    if not 0 <= row < spec["rows"]:
                        raise ValueError("invalid published sample row")
                    offset = shard["data_start"] + spec["data_offsets"][0] + row * width
                    data = os.pread(file.fileno(), width, offset)
                    if len(data) != width:
                        raise ValueError("short checkpoint sample")
                    digest.update(row.to_bytes(8, "little") + data)
            if digest.hexdigest() != sample[f"{kind}_sample_bytes_sha256"]:
                raise ValueError(
                    f"checkpoint content differs from owner at layer {layer}"
                )


@triton.jit
def _scatter_rows(
    weight,
    scales,
    ids,
    out,
    lo,
    hi,
    stride_w,
    stride_s,
    stride_t,
    stride_h,
    local_heads: tl.constexpr,
    packed: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids + row).to(tl.int64)
    owned = (idx >= lo) & (idx < hi)
    cols = tl.arange(0, 256)
    values = tl.load(weight + row * stride_w + cols, mask=owned, other=0.0)
    scale = tl.load(scales + row * stride_s + cols // 32, mask=owned, other=0)
    dst = out + (row // local_heads) * stride_t + (row % local_heads) * stride_h
    if packed:
        tl.store(dst + cols, values.to(tl.uint8, bitcast=True))
        tl.store(dst + 256 + cols // 32, scale, mask=cols % 32 == 0)
    else:
        factor = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(dst + cols, (values.to(tl.float32) * factor).to(tl.bfloat16))


class VllmRemoteReader:
    def __init__(self):
        from shared_model_adapter import ReaderRuntime

        if os.environ.get("ENGRAM_SHARED_PREFETCH") == "1":
            raise RuntimeError("vLLM prototype requires ENGRAM_SHARED_PREFETCH=0")
        verify_checkpoint_samples()
        self.runtime = ReaderRuntime()
        if self.runtime.storage_layout != "row264":
            self.runtime.close()
            raise RuntimeError("vLLM prototype requires packed row264 tables")
        self.consumed = torch.cuda.Event()
        self.inflight = False

    def lookup(self, table, indices, out, background=False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("remote lookup must run at a vLLM eager boundary")
        if table.dim != 256 or table.block_size != 32:
            raise ValueError("unsupported remote Engram row format")
        if out.dtype not in (torch.bfloat16, torch.uint8) or out.ndim != 3:
            raise ValueError("expected 3-D BF16 or packed-byte output")
        if out.shape[:2] != (indices.shape[0], table.part_n_hash_cols):
            raise ValueError("output shape does not match local Engram heads")
        start, heads = table.head_start, table.part_n_hash_cols
        local = indices[:, start : start + heads]
        if local.shape[1] < heads:
            local = F.pad(local, (0, heads - local.shape[1]), value=-1)
        ids = local.to(dtype=torch.int64).contiguous().view(-1)
        # The native GET validates every input ID before applying rank bounds.
        # vLLM uses negative IDs for padding, so give the transport a valid
        # placeholder and retain the original IDs for the final ownership mask.
        read_ids = torch.where((ids >= 0) & (ids < table.num_embeddings), ids, 0)
        stream = torch.cuda.current_stream()
        if self.inflight:
            stream.wait_event(self.consumed)
        _, weight, scale = self.runtime.fetch(
            read_ids, table.rdma_layer_id, table.vocab_start_idx, table.vocab_end_idx
        )
        _scatter_rows[(ids.numel(),)](
            weight,
            scale,
            ids,
            out,
            table.vocab_start_idx,
            table.vocab_end_idx,
            weight.stride(0),
            scale.stride(0),
            out.stride(0),
            out.stride(1),
            heads,
            out.dtype == torch.uint8,
            num_warps=4,
        )
        self.runtime.record_consumed(table.rdma_layer_id)
        self.consumed.record(stream)
        self.inflight = True


_reader = None


def get_reader():
    global _reader
    if _reader is None:
        _reader = VllmRemoteReader()
    return _reader


def close_reader():
    """Retire GPU/NIC work before releasing the owner lease."""
    global _reader
    if _reader is not None:
        _reader.runtime.close()
        _reader = None
