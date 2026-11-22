# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in routed-expert load counting for DeepSeek-V4.1 on the AITER MegaMoE
experts (`VLLM_AKR_EXPERT_COUNTS=<dir>`; unset, nothing imports this module).

Every MoE layer adds its routed (token, expert) pairs, the logical ids the
router chose (before any placement remap or shared-expert append), into one
persistent int64 GPU buffer `[2 kinds, 64 layers, 384 experts]` with one Triton
launch, so it is CUDA graph safe and is replayed with the graphs. Rows that the
forward context marks as CUDA graph or DP padding and ids < 0 are not counted.
The kind (0: decode-only step, 1: a step that carries prefill tokens) is a
device scalar that `ExpertCountsModelState` writes before every real step.

The first real step zeroes the buffer (drops profiling, warm-up and capture
traffic) and starts a thread that copies it to the host on a side stream every
`DUMP_INTERVAL_S` and writes `<dir>/expert_counts_dp<rank>.npz` (also at exit).
`akr_eplb_plan` turns the files of all DP ranks into a placement plan.
"""

import atexit
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import (
    get_moe_padding_mask,
)
from vllm.models.deepseek_v41.amd.akr_eplb import layer_index
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

logger = init_logger(__name__)

MAX_LAYERS = 64
MAX_EXPERTS = 384
DUMP_INTERVAL_S = 60.0
_KIND_STRIDE = MAX_LAYERS * MAX_EXPERTS
_BLOCK = 1024


# no integer specialization: a 1-row capture must not compile a new variant
@triton.jit(do_not_specialize=["n", "stride_row", "base"])
def _count_kernel(
    ids_ptr,
    pad_ptr,
    kind_ptr,
    buf_ptr,
    n,
    stride_row,
    base,
    K: tl.constexpr,
    HAS_PAD: tl.constexpr,
    KIND_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n * K
    r = o // K
    e = tl.load(ids_ptr + r * stride_row + o % K, mask=m, other=-1).to(tl.int64)
    ok = m & (e >= 0)
    if HAS_PAD:
        ok = ok & (tl.load(pad_ptr + r, mask=m, other=1) == 0)
    kind = tl.load(kind_ptr).to(tl.int64)
    tl.atomic_add(
        buf_ptr + kind * KIND_STRIDE + base + e,
        tl.full([BLOCK], 1, tl.int64),
        mask=ok,
        sem="relaxed",
    )


class _State:
    def __init__(self) -> None:
        self.out_dir = ""
        self.dp_rank = 0
        # prefix -> (slot, n_experts, topk), None for a layer that is not counted
        self.layers: dict[str, tuple[int, int, int] | None] = {}
        # slot -> (layer id, n_experts, topk)
        self.slots: dict[int, tuple[int, int, int]] = {}
        self.buf: torch.Tensor | None = None
        self.kind: torch.Tensor | None = None
        self.host: torch.Tensor | None = None
        self.stream: torch.cuda.Stream | None = None
        self.steps = [0, 0]
        self.started = 0.0
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.lock = threading.Lock()


_S = _State()


def configure(out_dir: str, dp_rank: int) -> None:
    _S.out_dir = out_dir
    _S.dp_rank = dp_rank


def _alloc(device: torch.device) -> None:
    _S.buf = torch.zeros(2 * _KIND_STRIDE, dtype=torch.int64, device=device)
    _S.kind = torch.zeros(1, dtype=torch.int32, device=device)
    _S.host = torch.empty(2 * _KIND_STRIDE, dtype=torch.int64, pin_memory=True)


def _register(prefix: str, n_experts: int, topk: int) -> tuple[int, int, int] | None:
    layer = layer_index(prefix)
    used = {v[0] for v in _S.layers.values() if v is not None}
    st = None
    if 0 <= layer < MAX_LAYERS and layer not in used and n_experts <= MAX_EXPERTS:
        st = (layer, n_experts, topk)
        _S.slots[layer] = st
    else:
        logger.warning("expert counts: not counting %s (%d experts)", prefix, n_experts)
    _S.layers[prefix] = st
    return st


def count(prefix: str, n_experts: int, topk_ids: torch.Tensor) -> None:
    """Add one MoE layer's routed pairs (`topk_ids [rows, topk]`, logical ids)
    to its counters."""
    if prefix in _S.layers:
        st = _S.layers[prefix]
    else:
        st = _register(prefix, n_experts, topk_ids.shape[1])
    if st is None or topk_ids.shape[0] == 0:
        return
    if _S.buf is None:
        if torch.cuda.is_current_stream_capturing():
            return  # the eager profile run allocates first
        _alloc(topk_ids.device)
    assert _S.buf is not None and _S.kind is not None
    slot, _, topk = st
    m = topk_ids.shape[0]
    pad = get_moe_padding_mask(m)
    if topk_ids.stride(1) != 1:
        topk_ids = topk_ids.contiguous()
    _count_kernel[(triton.cdiv(m * topk, _BLOCK),)](
        topk_ids,
        topk_ids if pad is None else pad,
        _S.kind,
        _S.buf,
        m,
        topk_ids.stride(0),
        slot * MAX_EXPERTS,
        K=topk,
        HAS_PAD=pad is not None,
        KIND_STRIDE=_KIND_STRIDE,
        BLOCK=_BLOCK,
    )


def step_begin(has_prefill: bool) -> None:
    """Before every real step's forward: select the decode or prefill
    counters."""
    if _S.buf is None:
        _alloc(torch.device("cuda", torch.accelerator.current_device_index()))
    assert _S.buf is not None and _S.kind is not None
    if _S.thread is None:
        _S.buf.zero_()
        _S.started = time.time()
        _S.stream = torch.cuda.Stream(_S.buf.device)
        _S.thread = threading.Thread(
            target=_dump_loop, name="akr_expert_counts", daemon=True
        )
        _S.thread.start()
        atexit.register(_at_exit)
        logger.info(
            "expert counts: counting %d MoE layers into %s", len(_S.slots), _S.out_dir
        )
    kind = int(has_prefill)
    _S.kind.fill_(kind)
    _S.steps[kind] += 1


def snapshot() -> dict[str, np.ndarray]:
    """The counters as the arrays of the dump file. Backbone layers (the
    largest expert count) are `counts*`, the others (the DSpark draft)
    `draft_counts*`."""
    assert _S.buf is not None and _S.host is not None and _S.stream is not None
    # inference tensors (made under the runner's inference mode); the stream
    # context also selects the buffer's device in the dump thread
    with torch.inference_mode(), torch.cuda.stream(_S.stream):
        _S.host.copy_(_S.buf, non_blocking=True)
        _S.stream.synchronize()
    c = _S.host.numpy().reshape(2, MAX_LAYERS, MAX_EXPERTS).copy()
    slots = sorted(_S.slots)
    n_max = max((_S.slots[s][1] for s in slots), default=0)
    out: dict[str, np.ndarray] = {}
    for key, sel in (
        ("", [s for s in slots if _S.slots[s][1] == n_max]),
        ("draft_", [s for s in slots if _S.slots[s][1] != n_max]),
    ):
        n = _S.slots[sel[0]][1] if sel else 0
        topk = _S.slots[sel[0]][2] if sel else 1
        dec, pre = c[0][sel, :n], c[1][sel, :n]
        out[key + "counts"] = dec + pre
        out[key + "counts_decode"] = dec
        out[key + "counts_prefill"] = pre
        out[key + "layer_ids"] = np.array(sel, dtype=np.int64)
        out[key + "topk"] = np.int64(topk)
        out[key + "tokens_decode"] = dec.sum(1) // topk
        out[key + "tokens_prefill"] = pre.sum(1) // topk
    out["steps_decode"] = np.int64(_S.steps[0])
    out["steps_prefill"] = np.int64(_S.steps[1])
    out["dp_rank"] = np.int64(_S.dp_rank)
    out["elapsed_s"] = np.float64(time.time() - _S.started)
    return out


def dump() -> str | None:
    """Write `<dir>/expert_counts_dp<rank>.npz` (atomically); None before the
    first real step."""
    if _S.thread is None or not _S.out_dir:
        return None
    with _S.lock:
        data = snapshot()
        os.makedirs(_S.out_dir, exist_ok=True)
        path = os.path.join(_S.out_dir, f"expert_counts_dp{_S.dp_rank}.npz")
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            np.savez(f, **data)
        os.replace(tmp, path)
    return path


def _dump_loop() -> None:
    while not _S.stop.wait(DUMP_INTERVAL_S):
        try:
            dump()
        except Exception as e:
            logger.warning("expert counts: dump failed: %s", e)


def _at_exit() -> None:
    _S.stop.set()
    try:
        dump()
    except Exception as e:
        logger.warning("expert counts: final dump failed: %s", e)


class ExpertCountsModelState(DeepseekV41ModelState):
    """`DeepseekV41ModelState` that also tells the counters whether the step
    carries prefill tokens."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        configure(
            envs.VLLM_AKR_EXPERT_COUNTS,
            vllm_config.parallel_config.data_parallel_rank,
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        super().preprocess_state(
            input_batch, block_tables, kv_cache_config, num_computed_tokens
        )
        step_begin(input_batch.has_prefill)
