# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for the specialized decode.

Small-batch decode is bound by HBM bandwidth, and each kernel that starts on weights
not yet in L2 stalls on DRAM. Some kernels leave bandwidth idle, such as the attention
core and the all-reduces. At fixed points of each layer, this module launches a small
kernel (`cutedsl/common/prefetch.py`) on a side stream. It pulls the byte ranges of
the weights the next kernels read into L2, so that they start warm.

Prefetch points (`fork(point, layer, batch)`), each with its batches per TP:

- `oproj_down`: after the o-projection (B3-7), the MoE front's packed
  router + latent-down rows. Not at B8 (TP8 and TP16), where an A/B of the front
  favored running without it.
- `attention_moe`: at TP8 only, after the KDA input projection (B5-8) or the MLA
  core (B8), this layer's o-projection, the same front rows, the shared expert's
  weights and this rank's shard of the latent up projection (the MoE tail's weights,
  which otherwise stream from HBM on the critical path beside the next-attention
  prefetch). GB300 TP8 K3.1 DSpark k=4 R1 (B5) 32K step, ms: front + shared 10.28,
  + up projection 10.03, + o-projection 9.89.
- `next_attention`: once the routed experts are launched, the next layer's
  attention weights, capped in MiB per batch (`NEXT_ATTENTION`). Not above B8,
  where the HBM-bound experts lose more bandwidth than the attention gains.

At the `EXPERTS_AHEAD` batches the next layer's attention weights are not a side
stream launch: the routed experts' producers prefetch them (`ahead`), each CTA its
share, once its last weight stage is issued. The experts' drain (FC2 hand-off,
finalize) leaves HBM idle for ~10 us, and no extra CTAs hold SMs the WaveMoeTail
clusters need when the experts exit.

The front and shared-expert ranges are on the critical path. They launch with the
highest stream priority and the maximum shared-memory carveout, which needs a driver
launch because CuTe drops both attributes.

Each fork records an event on the main stream that the side stream waits on, so a
prefetch starts only after the work it overlaps. `join()` at the end of the step
makes the main stream wait for the side stream, so a captured graph always rejoins
it. This lives in `runtime/` rather than `cutedsl/` because it is host-side
orchestration across layers: a stream, events, and per-(point, layer, batch) tables
built from the model's layers.
"""

import ctypes
from functools import partial

import cuda.bindings.driver as cuda
import torch

from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute

from ..cutedsl.common.host import _checked, resolve_kernel
from ..cutedsl.common.policy import batch_table
from ..cutedsl.common.prefetch import compile_kernel
from ..cutedsl.moe.experts import PREFETCH_ROWS

# Per-TP batch -> MiB cap (or cap.carveout) of the next layer's attention weights,
# prefetched once the routed experts finish.
# Not at B>8: the HBM-bound experts lose more bandwidth than the attention gains.
NEXT_ATTENTION = {8: "1:96.100,2-3:96,4-7:64,8:32", 16: "1-8:1024"}
# Per-TP batch -> MiB cap of the next layer's attention weights that the routed
# experts prefetch in-kernel instead (``PrefetchController.ahead``); these batches
# take no ``next_attention`` launch. GB300 TP8 K3.1 DSpark k=4 step, 32K, ms
# (side-stream launch after the experts -> in-kernel): R1 (B5) 10.03 -> 9.73 (both with
# the up projection in ``attention_moe``); B20 / B60 with 112 MiB: 16.31 / 28.47 ->
# 16.43 / 28.83 (the experts' drain is too short for it): none.
# The table starts with this layer's MoE tail up-projection shard (read right after the
# experts). K3.1 DSpark k=4 DCP8 R1 136.5K step, ms, cap 64 / 32 / 16 / 8 MiB (+ up
# shard): 10.18 / 10.01 / 9.94 / 9.99; no ahead and no next_attention: 10.12. Large
# ahead prefetches still in flight when the experts exit delay the MoE tail's first
# loads (its latent and weights land ~0.4 us later per layer, in-kernel stamps).
EXPERTS_AHEAD = {8: "4-7:16", 16: ""}


def attention_weights(attention):
    """Next-attention weights in their consumption order."""
    packed = getattr(attention, "packed_projection", None)
    if packed is not None:  # K3.1 DSA
        layer, wq = attention.mla_attn.mla_attn, attention.dsa_indexer.wq_index
        query = (packed, attention.q_b_proj.weight, layer.W_UK_T, wq.weight)
        return (*query, layer.W_UV, attention.o_proj.weight)
    names = (
        *("in_proj_qkvgfab", "fused_qkv_a_g_proj", "fused_qkv_a_proj", "q_proj"),
        *("kv_a_proj_with_mqa", "f_b_proj", "q_b_proj", "g_proj"),
    )
    weights = [getattr(getattr(attention, n, None), "weight", None) for n in names]
    weights += [getattr(attention, "W_UK_T", None), getattr(attention, "W_UV", None)]
    weights = [w for w in weights if isinstance(w, torch.Tensor)]
    return (*weights, attention.o_proj.weight)


def weight_ranges(weights, byte_cap=0):
    """(address, bytes) of each weight's span in order, the total capped at
    `byte_cap` bytes (0: uncapped)."""
    ranges, remaining = [], byte_cap
    for weight in weights:
        if weight.numel() == 0:
            continue
        if weight.data_ptr() % 16:  # the kernels walk 16-byte aligned chunks
            raise ValueError("L2 prefetch ranges must start 16-byte aligned")
        span = 1 + sum((n - 1) * s for n, s in zip(weight.shape, weight.stride()))
        size = span * weight.element_size()
        if byte_cap:
            size = min(size, remaining) // 16 * 16
            remaining -= size
        if size:
            ranges.append((weight.data_ptr(), size))
        if byte_cap and remaining < 16:
            break
    return ranges


class PreparedWeightRanges:
    """One owner retains weights, immutable range table, executable and launch ABI."""

    def __init__(self, weights, *, byte_cap=0, grid=32, carveout=None, priority=False):
        self.weights = weights = tuple(weights)
        device = weights[0].device
        ranges = weight_ranges(weights, byte_cap)
        self.table = torch.tensor(ranges, dtype=torch.int64, device=device)
        self.compiled = compile_kernel(len(ranges), grid, carveout, device.index)
        self.config = None
        if not priority and carveout is None:
            self.views = to_cute(self.table, 16)
            return
        # Launch priority / carveout need a driver launch (CuTe drops both).
        self.executor, self.function = resolve_kernel(self.compiled, device.index)
        ids, self.attributes = cuda.CUlaunchAttributeID, []
        if priority:  # the greatest stream priority
            top = cuda.CUlaunchAttribute()
            top.id = ids.CU_LAUNCH_ATTRIBUTE_PRIORITY
            top.value.priority = _checked(cuda.cuCtxGetStreamPriorityRange())[1]
            self.attributes.append(top)
        if carveout is not None:
            smem = cuda.CUlaunchAttribute()
            smem.id = ids.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            smem.value.sharedMemCarveout = carveout
            self.attributes.append(smem)
        config = self.config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
        config.sharedMemBytes, config.attrs = 0, self.attributes
        config.numAttrs = len(self.attributes)
        self.pointer = ctypes.c_void_p(self.table.data_ptr())
        self.arguments = (ctypes.c_void_p * 1)(ctypes.addressof(self.pointer))
        self.address = ctypes.addressof(self.arguments)

    def launch(self, stream):
        if self.config is None:
            self.compiled(self.views, stream)
        else:
            self.config.hStream = stream
            _checked(cuda.cuLaunchKernelEx(self.config, self.function, self.address, 0))


class PrefetchController:
    """Prefetch launches per (point, layer, batch) on one side stream."""

    def __init__(self, runtime, layers, packed):
        self.device, tp = runtime.device, runtime.tp
        self.stream = torch.cuda.Stream(device=self.device)
        self.events, self.pending, self.tables = {}, False, {}
        self.ahead_tables: dict[tuple[int, int], torch.Tensor] = {}
        urgent = partial(PreparedWeightRanges, carveout=100, priority=True)
        ahead = batch_table(EXPERTS_AHEAD[tp])
        following = batch_table(NEXT_ATTENTION[tp])
        following = {b: cap for b, cap in following.items() if b not in ahead}
        for position, layer in enumerate(layers):
            if not layer.is_moe_layer:
                continue
            index, moe, attention = layer.layer_idx, layer.mlp, layer.self_attn
            up: tuple = ()
            # The front GEMM reads [router shard; down shard] from the packed copy.
            ranges = urgent((packed[index],), grid=8)
            self._add("oproj_down", index, range(3, 8), ranges)
            if tp == 8:
                shared = moe.shared_experts
                shared = shared.gate_up_proj.weight, shared.down_proj.weight
                # The MoE tail's K-split up projection reads this rank's rows.
                up_weight = moe.routed_expert_up_proj.weight
                rows = up_weight.shape[0] // tp
                up = (up_weight[runtime.rank * rows : (runtime.rank + 1) * rows],)
                # This layer's o-projection first: it runs right after the core.
                weights = (attention.o_proj.weight, packed[index], *shared, *up)
                ranges = urgent(weights, grid=32)
                batches = range(5, 9) if hasattr(attention, "in_proj_qkvgfab") else (8,)
                self._add("attention_moe", index, batches, ranges)
            if position + 1 < len(layers):
                weights = attention_weights(layers[position + 1].self_attn)
                self._add_capped("next_attention", index, weights, following)
                for batch, cap in ahead.items():
                    self._add_ahead(index, batch, weights, cap, up)

    def _add_ahead(self, layer, batch, weights, cap, first=()):
        ranges = weight_ranges(first) + weight_ranges(weights, cap * 1024**2)
        ranges = ranges[:PREFETCH_ROWS]
        table = torch.zeros((PREFETCH_ROWS, 2), dtype=torch.int64, device=self.device)
        if ranges:
            table[: len(ranges)] = torch.tensor(ranges, dtype=torch.int64)
        self.ahead_tables[layer, batch] = table

    def ahead(self, layer, batch):
        """The [PREFETCH_ROWS, 2] (address, bytes) table of the next layer's attention
        weights the routed experts of (layer, batch) prefetch in-kernel, or None
        (`EXPERTS_AHEAD`)."""
        return self.ahead_tables.get((layer, batch))

    def _add_capped(self, point, layer, weights, table):
        groups: dict[tuple, list[int]] = {}
        for batch, spec in table.items():
            cap, carveout = spec if isinstance(spec, tuple) else (spec, None)
            if cap:
                groups.setdefault((cap, carveout), []).append(batch)
        for (cap, carveout), batches in groups.items():
            ranges = PreparedWeightRanges(
                weights, byte_cap=cap * 1024**2, grid=32, carveout=carveout
            )
            self._add(point, layer, batches, ranges)

    def _add(self, point, layer, batches, ranges):
        key = point, layer
        if key not in self.events:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.events[key] = event
        for batch in batches:
            self.tables[point, layer, batch] = ranges

    def fork(self, point, layer, batch):
        ranges = self.tables.get((point, layer, batch))
        if ranges is None:
            return False
        event = self.events[point, layer]
        event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(event)
        self.pending = True
        ranges.launch(cuda.CUstream(self.stream.cuda_stream))
        return True

    def join(self):
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False
