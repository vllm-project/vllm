# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental bridge from vLLM Engram lookup to the NIXL GET shim."""

import atexit
import base64
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.distributed.engram_share.provider import EngramTable
from vllm.distributed.engram_share.rdma.runtime import Context, OwnerClient


def verify_checkpoint_samples(checkpoint_path):
    """Reject mismatched owner weights using published boundary-row samples."""
    from vllm.distributed.engram_share.rdma.owner import inspect_checkpoint

    ready = json.loads(Path(os.environ["ENGRAM_SHARED_READY"]).read_text())
    samples = ready.get("sample_validation")
    if not isinstance(samples, list) or {s.get("layer") for s in samples} != {1, 14}:
        raise ValueError("owner did not publish the required checkpoint samples")
    try:
        model = Path(checkpoint_path).resolve(strict=True)
        configured = Path(os.environ["ENGRAM_SHARED_MODEL"]).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError("remote Engram requires a local engine checkpoint") from exc
    if not model.is_dir() or model != configured:
        raise ValueError("engine checkpoint differs from ENGRAM_SHARED_MODEL")
    manifest = inspect_checkpoint(model)
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


# Failed retirement must keep buffers and the owner lease alive.
_quarantined = []


class ReaderRuntime:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        config = json.loads(Path(os.environ["ENGRAM_SHARED_READY"]).read_text())
        self.client = OwnerClient(
            config["host"],
            config["port"],
            config["checkpoint_id"],
            config["generation"],
        )
        self.lease = self.client.attach()
        self.context = None
        self.closed = False
        try:
            self._prepare()
        except BaseException:
            try:
                if self.context is not None:
                    self.context.close()
                self.lease.release()
            except BaseException:
                _quarantined.append(self)
            raise
        atexit.register(self.close)

    def _prepare(self):
        md = self.lease.metadata
        self.metadata = md
        self.storage_layout = md.get("storage_layout")
        if self.storage_layout != "row264":
            raise ValueError("remote Engram requires row264 owner tables")
        model = self.checkpoint_path
        for name, field in (
            ("config.json", "config_sha256"),
            ("model.safetensors.index.json", "index_sha256"),
        ):
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != md[field]:
                raise ValueError(
                    "shared owner/serving checkpoint identity mismatch: " + name
                )
        cfg = json.loads((model / "config.json").read_text())
        rows = cfg.get("text_config", cfg)["engram_num_embeddings"]
        specs = md.get("table_specs", ())
        expected = {
            "weight": {"offset": 0, "bytes_per_row": 256, "dtype": "F8_E4M3"},
            "scale": {"offset": 256, "bytes_per_row": 8, "dtype": "F8_E8M0"},
        }
        if (
            len(specs) != 2
            or tuple(md["lengths"]) != tuple(int(n) * 264 for n in rows)
            or any(
                s.get("layer") != layer or s.get("component_layout") != expected
                for s, layer in zip(specs, (1, 14))
            )
        ):
            raise ValueError("owner row264 table layout mismatch")
        self.max_queries = int(os.environ.get("ENGRAM_SHARED_MAX_QUERIES", "196608"))
        if self.max_queries <= 0:
            raise ValueError("ENGRAM_SHARED_MAX_QUERIES must be positive")
        self.p_off = 64
        self.dump = ((self.p_off + self.max_queries * 264 + 127) // 64) * 64
        gpu = torch.cuda.current_device()
        self.arena = torch.empty(self.dump + 64, dtype=torch.uint8, device="cuda")
        self.stats = torch.zeros((8,), dtype=torch.uint64, device="cuda")
        port = int(os.environ["ENGRAM_SHARED_READER_PORT_BASE"]) + gpu * 4
        self.context = Context(os.environ["ENGRAM_SHARED_LIBRARY"], gpu, port)
        if not self.context.supports_row264:
            raise RuntimeError("selected NIXL runtime lacks row264 capability")
        self.context.register([self.arena], host=False)
        self.context.connect_raw(
            base64.b64decode(md["native_md"], validate=True),
            md["addresses"],
            md["lengths"],
            md["owner_gpu"],
        )

    def fetch(self, indices, layer, lo, hi):
        if self.closed:
            raise RuntimeError("shared Engram reader is closed")
        ids = indices.reshape(-1).contiguous()
        n = ids.numel()
        if n > self.max_queries:
            raise ValueError("shared Engram scratch capacity exceeded")
        index = (1, 14).index(layer)
        self.context.fetch(
            ids,
            n,
            self.p_off,
            0,
            self.dump,
            512,
            self.stats,
            weight_index=index,
            scale_index=index,
            row_lo=lo,
            row_hi=hi,
        )
        torch._assert_async(
            self.stats.view(torch.int64)[0] == 0, "shared Engram GET failed"
        )
        raw = self.arena[self.p_off : self.p_off + n * 264].view(n, 264)
        return ids, raw[:, :256].view(torch.float8_e4m3fn), raw[:, 256:]

    def record_consumed(self, layer):
        self.context.record_consumed()

    def close(self):
        if not self.closed:
            self.context.close()
            self.lease.release()
            self.closed = True


class VllmRemoteReader:
    """Own the RDMA runtime, scratch and lookup ordering for one worker."""

    def __init__(self, checkpoint_path):
        if os.environ.get("ENGRAM_SHARED_PREFETCH") == "1":
            raise RuntimeError("vLLM prototype requires ENGRAM_SHARED_PREFETCH=0")
        verify_checkpoint_samples(checkpoint_path)
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        self.runtime = ReaderRuntime(self.checkpoint_path)
        try:
            if self.runtime.storage_layout != "row264":
                raise RuntimeError("vLLM prototype requires packed row264 tables")
            self.consumed = torch.cuda.Event()
        except BaseException:
            try:
                self.runtime.close()
            except BaseException:
                _quarantined.append(self.runtime)
            raise
        self.inflight = False
        self.last_stream = None
        self.captured_stream = None

    def lookup(
        self,
        table: EngramTable,
        indices: torch.Tensor,
        out: torch.Tensor,
        background: bool = False,
    ) -> None:
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not getattr(table, "rdma_full_decode_graph", False):
            raise RuntimeError("remote lookup must run at a vLLM eager boundary")
        if table.dim != 256 or table.block_size != 32:
            raise ValueError("unsupported remote Engram row format")
        if out.dtype not in (torch.bfloat16, torch.uint8) or out.ndim != 3:
            raise ValueError("expected 3-D BF16 or packed-byte output")
        if out.shape[-1] != (264 if out.dtype == torch.uint8 else 256):
            raise ValueError("output row width does not match its dtype")
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
        if (
            self.captured_stream is not None
            and self.captured_stream != stream.cuda_stream
        ):
            raise RuntimeError(
                "captured remote reader must stay on its prefetch stream"
            )
        if self.inflight:
            if capturing and self.last_stream != stream.cuda_stream:
                raise RuntimeError("warm remote reader on capture stream")
            # An event recorded during capture cannot be waited on by unrelated
            # eager work. The same stream already orders reuse of the arena.
            if not capturing and self.last_stream != stream.cuda_stream:
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
        self.last_stream = stream.cuda_stream
        if capturing:
            self.captured_stream = stream.cuda_stream

    def close(self) -> None:
        """Retire GPU and NIC work before returning the owner lease."""
        self.runtime.close()


_reader = None


def get_reader(checkpoint_path=None):
    global _reader
    if _reader is None:
        if checkpoint_path is None:
            raise RuntimeError("remote reader must be initialized by the model loader")
        _reader = VllmRemoteReader(checkpoint_path)
    elif checkpoint_path is not None and (
        Path(checkpoint_path).resolve(strict=True) != _reader.checkpoint_path
    ):
        raise ValueError("remote reader belongs to a different engine checkpoint")
    return _reader


def close_reader():
    """Retire GPU/NIC work before releasing the owner lease."""
    global _reader
    if _reader is not None:
        _reader.close()
        _reader = None
