# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-lane Engram reader for the experimental GPU GET transport."""

import atexit
import base64
import hashlib
import json
import os
from pathlib import Path

import torch

from vllm.distributed.engram_rdma.control import OwnerClient
from vllm.distributed.engram_rdma.transport import Context

# Failed retirement must keep buffers and the owner lease alive.
_quarantined = []


class ReaderRuntime:
    def __init__(self):
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
        model = Path(os.environ["ENGRAM_SHARED_MODEL"])
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
            row264=True,
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
