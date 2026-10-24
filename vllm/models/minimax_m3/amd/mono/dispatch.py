# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether a step runs on the mono kernels, and the runner it runs on.

The model holds one of these and its sparse layers ask it, each step, whether to
take the kernels or their own modules. The first of them calls ``begin``, which
settles the step for all of them; a step the kernels cannot serve simply leaves
every layer on the original path. There is no switch to set: a deployment the
kernels cannot serve at all makes the first attempt raise ``MonoUnsupported``,
which turns this off for the process.

The per-step conditions are all the ones a decode batch can change: token count,
no prefill mixed in, a single microbatch, no piecewise graph (mono is one launch
a layer, so a piecewise graph would split it) and the tables covering every row.
Everything fixed by the deployment -- TP, pipelining, CU count, KV cache layout,
quantization and weight layouts -- is checked once when the runner is built.

Every condition has to be decided from TP-replicated state alone. The kernels
all-reduce in-kernel, so ranks that disagreed would not produce wrong numbers,
they would wait on each other forever.

A full graph's padding rows are served rather than refused, since they are what a
captured decode replays: their slots are ``PAD_SLOT_ID``, so K1 skips their cache
writes, and padding is in whole requests, so the block table and sequence lengths
cover every row.
"""

import torch

from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.mono.config import (
    MAX_CONTEXT,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    MonoUnsupported,
)
from vllm.models.minimax_m3.amd.mono.runner import (
    N_DENSE,
    MonoDecodeRunner,
    token_rows,
)

logger = init_logger(__name__)


def _config_refusal() -> str | None:
    """Why this deployment cannot use mono at all, or None.

    Taken from the engine's config rather than the model, so a server mono cannot
    serve says so while it is starting instead of at its first decode. What the
    model itself has to satisfy is checked when the runner is built.
    """
    try:
        cfg = get_current_vllm_config()
    except Exception:
        # No engine around the model: nothing to refuse on, and the runner still
        # checks everything that decides whether its kernels can run.
        return None
    checks = (
        # Each replica holds the whole model and reduces within its own TP group,
        # so a second replica would route the MoE differently than K4 reduces.
        (
            cfg.parallel_config.data_parallel_size == 1,
            f"DP {cfg.parallel_config.data_parallel_size}",
        ),
        # The score region is sized for MAX_CONTEXT, and a longer context would
        # be scored past the end of it rather than merely more slowly.
        (
            cfg.model_config.max_model_len <= MAX_CONTEXT,
            f"max_model_len {cfg.model_config.max_model_len} > {MAX_CONTEXT}",
        ),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


class MonoDecode:
    """Owns the lazily built runner and the per-step decision."""

    def __init__(self, model) -> None:
        self.model = model
        self.runner: MonoDecodeRunner | None = None
        self.off = False
        self.off_reason = ""
        # settled by ``begin`` each step, read by the layers behind the first
        self.active = False
        self._fwd = None
        self._rows: tuple[torch.Tensor, torch.Tensor] | None = None
        self.rank = get_tensor_model_parallel_rank()
        layers = model.layers
        # None rather than a hard error: a config with no sparse layer here is
        # simply never eligible, and the runner is what reports why. Under
        # pipelining the slot can also hold a placeholder with no attention.
        self.attn = None
        if len(layers) > N_DENSE:
            self.attn = getattr(layers[N_DENSE], "self_attn", None)
        self.layer_name = getattr(self.attn, "layer_name", None)
        why = _config_refusal()
        if why is not None:
            self.off, self.off_reason = True, why
            logger.info("[mono] off: %s", why)
        self._attach(layers)

    def _attach(self, layers) -> None:
        """Give every sparse layer that can run on the kernels its slot in them.

        A slot is the layer's place in the chain the kernels ping-pong along, not
        its layer id. Layers that do not declare the attribute are left alone, so a
        model built from the original class keeps working.
        """
        for lid in range(N_DENSE, len(layers)):
            layer = layers[lid]
            if hasattr(layer, "mono_slot"):
                layer.mono = self
                layer.mono_slot = lid - N_DENSE

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None
        self.off = True
        self.off_reason = "closed"

    def _say(self, what: str) -> None:
        """Every step, from every rank, at debug level.

        Not deduplicated: while the path is being brought up, a step that says
        nothing is indistinguishable from one that was never offered. The rank is
        in the message because what a rank decided is only meaningful against what
        the others decided, and a step the kernels serve has to be unanimous.
        """
        logger.info("[mono][rank %d] %s", self.rank, what)

    def _declined(
        self, n: int, positions: torch.Tensor, residual: torch.Tensor | None
    ) -> str | None:
        """Why this step cannot run on the kernels, or None when it can."""
        if self.layer_name is None:
            return "no sparse layer at this position"
        # A sparse layer is never the model's first, so the dense layers have
        # already produced one; a step without one would mean the layer counts the
        # kernels are built around no longer hold.
        if residual is None:
            return "no residual at the first sparse layer"
        if not 1 <= n <= MAX_TOKENS:
            return f"{n} tokens, kernels serve 1..{MAX_TOKENS}"
        fwd = get_forward_context()
        if fwd.ubatch_slices is not None or not isinstance(fwd.attn_metadata, dict):
            return "microbatched step"
        if fwd.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return "piecewise cudagraph would split the layer launch"
        md = fwd.attn_metadata.get(self.layer_name)
        if md is None or md.decode is None:
            return "no sparse decode metadata"
        if md.num_prefills:
            return f"{md.num_prefills} prefill(s) in the step"
        # the kernels read a token's position and slot as an int64's low word
        if positions.dtype != torch.int64:
            return f"positions are {positions.dtype}"
        why = self._tables_fit(md, n)
        if why is not None:
            return why
        if self.runner is None:
            # The runner freezes each layer's index cache address into its
            # argument array, so it cannot be built before the caches the engine
            # will serve with exist.
            if self.attn is None or self.attn.kv_cache.numel() == 0:
                return "KV caches not bound yet (memory profiling)"
            # TODO: fix this correctly.
            blocks, want = self.attn.kv_cache.shape[0], md.decode.block_table.shape[1]
            if blocks < want:
                return (
                    f"KV cache of {blocks} blocks is the cudagraph memory "
                    f"profiler's, not the engine's {want} or more"
                )
            # It also allocates, syncs and exchanges IPC handles over the TP CPU
            # group, none of which a capturing stream tolerates. vLLM warms each
            # capture size up with an uncaptured run first, so the build lands
            # there.
            if torch.cuda.is_current_stream_capturing():
                return "cudagraph capture reached before the runner was built"
        return None

    @staticmethod
    def _tables_fit(md, n: int) -> str | None:
        """Whether the step's tables cover ``n`` rows in the layout K1 / K4 read.

        A graph's padding rows are real rows to the kernels, while an eager decode
        cuts the tables to the true batch and leaves the token axis padded, so the
        row counts are checked against the tokens actually launched.
        """
        decode = md.decode
        q = decode.decode_query_len
        if n % q:
            return f"{n} rows is not a multiple of {q} tokens a request"
        reqs = n // q
        bt, seq_lens = decode.block_table, decode.seq_lens
        if bt.dtype != torch.int32 or seq_lens.dtype != torch.int32:
            return f"block table {bt.dtype} / seq lens {seq_lens.dtype}, want int32"
        if bt.shape[0] < reqs or seq_lens.shape[0] < reqs:
            return (
                f"{reqs} requests but {bt.shape[0]} block table / "
                f"{seq_lens.shape[0]} seq len rows"
            )
        # bt_row(k) steps by exactly one row of bt_width int32s
        if bt.stride(0) != bt.shape[1]:
            return f"block table rows are {bt.stride(0)} apart, not {bt.shape[1]}"
        # The score region holds MAX_INDEX_BLOCKS scores a row, so a deployment
        # whose block table is wider has contexts the indexer cannot score. Taken
        # from the table's width rather than a sequence length, which would be a
        # device read, and it bounds every step the same way the config does.
        if bt.shape[1] > MAX_INDEX_BLOCKS:
            return f"{bt.shape[1]} blocks a request, scores hold {MAX_INDEX_BLOCKS}"
        slots = md.slot_mapping
        if slots.dtype != torch.int64:
            return f"slot mapping is {slots.dtype}, want int64"
        if slots.numel() < n:
            return f"slot mapping holds {slots.numel()} of {n} rows"
        return None

    def begin(
        self, rows: int, positions: torch.Tensor, residual: torch.Tensor | None
    ) -> bool:
        """Settle this step for every sparse layer, and cache what they share.

        The first sparse layer calls this, rather than the model's forward calling
        it, so the original loop carries no mono-specific code at all. One decision
        a step: the layers behind read ``active``, so they cannot disagree, and the
        row tables are built once instead of 57 times.
        """
        self.active = False
        # Repeated rather than said once at the moment it was decided: the reason
        # is settled during startup, thousands of lines before the first request.
        if self.off:
            self._say(f"off: {self.off_reason}")
            return False
        why = self._declined(rows, positions, residual)
        if why is not None:
            self._say(f"skip ({rows} rows): {why}")
            return False
        if self.runner is None:
            try:
                self.runner = MonoDecodeRunner(self.model)
            except MonoUnsupported as why:
                self.off_reason = str(why)
                self._say(f"OFF for the rest of the run: {why}")
                self.off = True
                return False
            self._say(f"ARMED for {len(self.runner.sparse)} sparse layers")
        self._fwd = get_forward_context()
        decode = self.runner.metadata_of(self._fwd, self.runner.sparse[0]).decode
        self._rows = token_rows(decode, rows)
        self.active = True
        self._say(f"serve {rows} rows")
        return True

    def run_layer(
        self,
        slot: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse layer ``slot`` of a step ``begin`` accepted."""
        return self.runner.run_layer(
            slot, self._fwd, positions, hidden_states, residual, self._rows
        )
