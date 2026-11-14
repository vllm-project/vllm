# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""FlashTopK: exact, deterministic top-k selection for learned sparse attention.

    selector = flashtopk.Selector(topk=2048, dtype=torch.float8_e4m3fn).plan(max_rows, max_kv)
    indices = selector.run(q, kv, weights, ks, ke, kv_scales)

or flashtopk.select(...) for the same with a cached selector. See topk.py for the score and the output order.
"""

from . import ops  # registers torch.ops.vllm_flashtopk.select
from .topk import DEFAULT_CHUNK_SIZE, Info, Selector, select, supported, windows_from_cu_seqlens, workspace_bytes

__all__ = ["DEFAULT_CHUNK_SIZE", "Info", "Selector", "ops", "select", "supported", "windows_from_cu_seqlens",
           "workspace_bytes"]
__version__ = "0.1.0"
