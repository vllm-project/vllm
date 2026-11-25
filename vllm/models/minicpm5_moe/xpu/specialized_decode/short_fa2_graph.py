# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One extra q1 Rows8 graph for the measured short-KV TP2 pre-O path."""

from collections import Counter
from weakref import ref

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

# The 1024-input/128-output real decode experiment covers this bound.
SHORT_KV_MAX = 1152


class ShortFa2Selection:
    def __init__(self):
        self.max_kv = None

    def prepare(self, input_batch, cg_mode, for_capture):
        self.max_kv = None
        real = input_batch.num_reqs
        if (
            for_capture
            or cg_mode != CUDAGraphMode.FULL
            or not 5 <= real <= 8
            or input_batch.num_tokens_after_padding != 8
            or input_batch.num_reqs_after_padding != 8
            or input_batch.num_tokens != real
            or input_batch.has_prefill
            or input_batch.num_draft_tokens
            or input_batch.is_prefilling_np.any()
            or not (input_batch.prefill_len_np > 0).all()
            or not (input_batch.num_scheduled_tokens == 1).all()
        ):
            return
        # Existing CPU upper bounds include the current scheduled query.
        # A conservative overestimate can only select the automatic graph.
        lengths = input_batch.seq_lens_cpu_upper_bound
        if lengths.device.type != "cpu" or lengths.numel() < real:
            return
        lengths = lengths[:real]
        if lengths.min().item() > 0:
            self.max_kv = int(lengths.max().item())


class _Rows8Fa2Graphs:
    def __init__(self, manager, desc, wrappers, selection):
        # Only the primary manager owns this object. The model state survives
        # removal of the profiling manager and its temporary KV pointers.
        self._manager = ref(manager)
        self._capture = type(manager).capture
        self._run_fullgraph = type(manager).run_fullgraph
        self.desc = desc
        self.wrappers = wrappers
        self.selection = selection
        self.short = None
        self.replays = Counter()
        self.max_kv = {}
        self.last_variant = None
        self.last_max_kv = None

    def capture(self, *args, **kwargs):
        manager = self._manager()
        self._capture(manager, *args, **kwargs)
        # ModelCudaGraphManager keeps capture and runtime attention buffers
        # alive through the runner. Both variants consume the same buffers.
        # The variant owns its output and captures its temporaries separately.
        short = ModelCudaGraphManager(
            manager.vllm_config, manager.device, manager.cudagraph_mode, {1: 8}
        )
        short._capture_descs = {CUDAGraphMode.FULL: [self.desc]}
        short.pool = torch.xpu.graph_pool_handle()
        short._capture_stream = manager._capture_stream
        before = tuple(wrapper.calls for wrapper in self.wrappers)
        # The stock capture helper chooses its own capture stream. Finish the
        # previous dummy writes before preparing the alternative graph.
        manager.capture_stream.synchronize()
        try:
            for wrapper in self.wrappers:
                wrapper.force_one = True
            short.capture(*args, **kwargs)
            short.capture_stream.synchronize()
        finally:
            for wrapper in self.wrappers:
                wrapper.force_one = False
        if not all(w.calls > n for w, n in zip(self.wrappers, before, strict=True)):
            raise RuntimeError(
                "The short MiniCPM5 graph did not exercise each routed FA2 layer"
            )
        self.short = short

    def run_fullgraph(self, desc):
        max_kv = self.selection.max_kv
        self.selection.max_kv = None
        if desc == self.desc and self.short is not None:
            variant = (
                "short" if max_kv is not None and max_kv <= SHORT_KV_MAX else "auto"
            )
            self.last_variant, self.last_max_kv = variant, max_kv
            self.replays[variant] += 1
            if max_kv is not None:
                self.max_kv[variant] = max(max_kv, self.max_kv.get(variant, 0))
            if variant == "short":
                return self.short.run_fullgraph(desc)
        return self._run_fullgraph(self._manager(), desc)


def attach_short_fa2_graph(manager, decode, selection):
    """Install the policy only on the eligible target manager instance."""
    if manager.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return
    rows8 = [
        desc
        for desc in manager._capture_descs.get(CUDAGraphMode.FULL, ())
        if (
            desc.num_tokens == desc.num_reqs == 8
            and desc.uniform_token_count == 1
            and desc.num_active_loras == 0
            and desc.num_ubatches == 1
        )
    ]
    wrappers = tuple(
        layer.pre_o_fa2 for layer in decode.layers.values() if not layer.dense
    )
    if len(rows8) != 1 or len(wrappers) != 27 or any(w is None for w in wrappers):
        return
    variants = _Rows8Fa2Graphs(manager, rows8[0], wrappers, selection)
    manager.short_fa2_graphs = variants
    manager.capture = variants.capture
    manager.run_fullgraph = variants.run_fullgraph
