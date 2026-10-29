# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which MiniMax-M3 forwards take the mono path (``VLLM_ROCM_MINIMAX_M3_MONO=1``).

The deployment is checked once (``_config_refusal``); a step is routed to the
kernels only when it is a pure decode of 1..MAX_TOKENS tokens whose tables the
kernels can read (``_step_refusal``). Every per-step condition is decided from
TP-replicated state: the kernels all-reduce in-kernel, so ranks that disagreed
would wait on each other forever.

The runner is built on the first eligible step, which under cudagraphs is the
warmup run before the first capture (its allocations and IPC handshake cannot be
captured), and rebuilt when vLLM re-binds the KV cache after its memory-profiling
capture round.
"""

from typing import Any, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.mono.check import forward_checked
from vllm.models.minimax_m3.amd.mono.config import (
    MAX_CONTEXT,
    MAX_TOKENS,
    SPARSE_BLOCK,
    TP,
    MonoUnsupported,
)
from vllm.models.minimax_m3.amd.mono.envs import envs
from vllm.models.minimax_m3.amd.mono.runner import MonoDecodeRunner, dense_prefix

logger = init_logger(__name__)


def _dense_prefix_ok(layers) -> bool:
    try:
        dense_prefix(layers)
    except MonoUnsupported:
        return False
    return True


def _separate_shared_ok(layers) -> bool:
    """Every MoE layer keeps its shared expert as a separate MLP (the kernels then
    run it from that MLP's MXFP8 weights, ``weights.mxfp8_a_tiles``)."""
    moes = [lyr.mlp for lyr in layers if lyr.is_moe_layer]
    return bool(moes) and all(
        not m.is_fused_shared_expert_enabled
        and getattr(m, "shared_experts", None) is not None
        for m in moes
    )


def _config_refusal(vllm_config: VllmConfig, model) -> str | None:
    """Why this deployment cannot use the kernels at all, or None."""
    pc = vllm_config.parallel_config
    cc = vllm_config.cache_config
    cg = vllm_config.compilation_config.cudagraph_mode
    max_len = vllm_config.model_config.max_model_len
    index_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("bf16")
    checks = (
        (pc.tensor_parallel_size == TP, f"TP {pc.tensor_parallel_size} != {TP}"),
        (pc.data_parallel_size == 1, "DP > 1"),
        (pc.pipeline_parallel_size == 1, "PP > 1"),
        (not pc.enable_expert_parallel, "EP"),
        (str(cc.cache_dtype).startswith("fp8"), f"kv cache {cc.cache_dtype}"),
        (index_dtype == "fp8", f"index cache {index_dtype}"),
        (cc.block_size == SPARSE_BLOCK, f"block size {cc.block_size}"),
        (max_len <= MAX_CONTEXT, f"max_model_len {max_len} > {MAX_CONTEXT}"),
        (cg is None or not cg.requires_piecewise_compilation(), "piecewise cudagraph"),
        (
            getattr(model, "is_fused_shared_expert_enabled", False)
            or _separate_shared_ok(model.layers),
            "shared expert neither fused nor a separate MLP",
        ),
        (_dense_prefix_ok(model.layers), "layer mix"),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


def _step_refusal(
    fwd: ForwardContext, layer_name: str, n: int, positions
) -> str | None:
    """Why this step cannot run on the kernels, or None when it can."""
    if not 1 <= n <= MAX_TOKENS:
        return f"{n} tokens"
    if positions.dtype != torch.int64:  # read as an int64's low word
        return f"positions {positions.dtype}"
    if fwd.ubatch_slices is not None:
        return "microbatched"
    if not isinstance(fwd.attn_metadata, dict) or not isinstance(
        fwd.slot_mapping, dict
    ):
        return "no per-layer metadata"
    md = cast(Any, fwd.attn_metadata.get(layer_name))
    if md is None or md.decode is None:
        return "no decode metadata"
    if md.num_prefills:
        return "prefill in the step"
    q = md.decode.decode_query_len
    if q < 1 or n % q:
        return f"{n} tokens for {q} per request"
    bt, seq_lens = md.decode.block_table, md.decode.seq_lens
    if bt.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return "tables not int32"
    if bt.shape[0] < n // q or seq_lens.shape[0] < n // q:
        return "tables shorter than the batch"
    if bt.stride(0) != bt.shape[1] or bt.stride(1) != 1:  # rows bt_width apart
        return "block table not row-contiguous"
    slots = fwd.slot_mapping.get(layer_name)
    if slots is None or slots.dtype != torch.int64 or slots.numel() < n:
        return "slot mapping"
    return None


class MonoDecode:
    """Routes supported decode steps of one ``MiniMaxM3Model`` to the kernels."""

    def __init__(self, model, vllm_config: VllmConfig) -> None:
        self._model = model
        self._config = vllm_config.model_config.hf_text_config
        self._runner: MonoDecodeRunner | None = None
        self._enabled = False
        self._slots_verified = False
        if not envs.MONO_ENABLE:
            return
        why = _config_refusal(vllm_config, model)
        if why is not None:
            logger.info("MiniMax-M3 mono decode off: %s", why)
            return
        attn0 = model.layers[dense_prefix(model.layers)].self_attn
        self._layer_name = attn0.layer_name
        self._index_prefix = attn0.indexer.index_cache.prefix
        self._enabled = True

    def _disable(self, why: str) -> None:
        logger.warning("MiniMax-M3 mono decode off: %s", why)
        self._enabled = False

    def _ensure_runner(self) -> bool:
        """Build (or rebuild after a KV cache re-bind) the runner outside capture."""
        if self._runner is not None and self._runner.stale():
            if torch.cuda.is_current_stream_capturing():
                return False
            self._runner.close()
            self._runner = None
            self._slots_verified = False
            logger.info("MiniMax-M3 mono decode: KV cache re-bound, rebuilding")
        if self._runner is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            self._runner = MonoDecodeRunner(self._model, self._config)
        except MonoUnsupported as why:
            self._disable(str(why))
            return False
        except Exception:
            # TP-uniform: every rank fails the same way, so all fall back
            logger.exception("MiniMax-M3 mono decode: runner build failed")
            self._enabled = False
            return False
        if self._runner.shared_fp8:
            copy = sum(lw.shared_bytes() for lw in self._runner.sparse)
            how = f"separate MXFP8 ({copy / 2**30:.2f} GiB tiled copy per rank)"
        else:
            how = "fused into the MXFP4 expert tables"
        logger.info("MiniMax-M3 mono decode on: shared expert %s", how)
        return True

    def _slots_shared(self, fwd: ForwardContext, n: int) -> bool:
        """The kernels insert K/V and the index key through one page-128 slot
        mapping, so the index cache must share the main cache's block ids.
        Checked once (a device sync)."""
        if self._slots_verified:
            return True
        slot_mapping = cast(dict[str, Any], fwd.slot_mapping)
        slots = slot_mapping[self._layer_name]
        index_slots = slot_mapping.get(self._index_prefix)
        if index_slots is None or not torch.equal(index_slots[:n], slots[:n]):
            self._disable("index cache does not share the KV cache's block ids")
            return False
        self._slots_verified = True
        return True

    def supports(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors,
        inputs_embeds: torch.Tensor | None,
    ) -> bool:
        if not self._enabled or intermediate_tensors is not None:
            return False
        fwd = get_forward_context()
        n = positions.numel()
        if _step_refusal(fwd, self._layer_name, n, positions) is not None:
            return False
        return self._ensure_runner() and self._slots_shared(fwd, n)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self._runner is not None
        if envs.MONO_CHECK:
            return forward_checked(self._runner, input_ids, positions, inputs_embeds)
        return self._runner.forward(input_ids, positions, inputs_embeds)
