# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode of the KIMI_K3_KV_BALANCED_MLA attention backend (``backend.py``) on
the KV-balanced kernel. It covers the target verify and the DSpark draft, with or
without DCP. Per forward, the first MLA layer of each decode shape plans the split-KV
work list from this rank's KV lengths, and every later layer of that forward reuses
it. Under DCP the kernel reads this rank's KV shard, masks causally in global
positions and returns the log2 LSE that the DCP combine merges.

The layers share one set of plan buffers per micro-batch lane, reserved with the
split-KV scratch (from the shared decode workspace) when the layers are
built, so the KV cache is sized around them. Every decode step runs on the kernel;
one it can't run raises instead of running some other decode.
"""

from typing import TYPE_CHECKING

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import max_decode_query_len
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonImpl,
        MLACommonMetadata,
    )
    from vllm.v1.attention.backend import AttentionLayer

    from .core import KvBalancedMla

logger = init_logger(__name__)

LATENT, ROPE = 512, 64


# The kernel per (heads, q, causal): the target verify and the DSpark draft.
_SHAPES: dict[tuple[int, int, bool], "KvBalancedMla"] = {}
# Per plan buffer (``work``): the (forward context, KV lengths, kernel) of the plan it
# holds. The context is held so that its identity can't be reused by a later forward.
_PLANNED: dict[int, tuple[object, int, "KvBalancedMla"]] = {}


class KvBalancedDecode:
    """The KV-balanced decode of one MLA layer with ``num_heads`` local heads."""

    def __init__(self, num_heads: int):
        from . import core

        config = get_current_vllm_config()
        # NIXL P/D resolves the KV interleave after the model is built: read it per
        # call.
        self.parallel_config = config.parallel_config
        dcp = config.parallel_config.decode_context_parallel_size
        # Under DCP the query is gathered over the ranks' heads.
        self.heads = num_heads * dcp
        self.max_q = max_decode_query_len(config)
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        if not core.supports(self.heads, self.max_q, device):
            raise ValueError(
                "KIMI_K3_KV_BALANCED_MLA needs more than 64 query rows per request "
                f"(heads x query length), got {self.heads} x {self.max_q}"
            )
        self.max_requests = config.scheduler_config.max_num_seqs
        max_planned = core.max_planned_requests(self.heads, self.max_q)
        if self.max_requests > max_planned:
            raise ValueError(
                f"KIMI_K3_KV_BALANCED_MLA plans at most {max_planned} requests per "
                f"step at {self.heads} heads x {self.max_q} query tokens; lower "
                f"max_num_seqs ({self.max_requests})"
            )
        clusters = core.max_active_clusters(device)
        self._key = ("kimi_k3_kv_balanced_mla", self.heads, self.max_q)
        self._counts_shape = (self.max_requests,)
        self._work_shape = (
            core.work_list_len(self.heads, self.max_q, self.max_requests, clusters),
        )
        self._workspace_bytes = core.workspace_bytes(
            self.heads, self.max_q, self.max_requests
        )
        logger.info_once(
            "Kimi K3 KV-balanced MLA decode: %.0f MiB of split-KV scratch for %d "
            "requests",
            self._workspace_bytes / 2**20,
            self.max_requests,
        )
        self._buffers()

    def _buffers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This layer shape's plan buffers (``counts``, ``work``) of the current
        micro-batch lane, and the split-KV scratch."""
        manager = current_workspace_manager()
        counts = manager.get_persistent(
            (*self._key, "counts"), self._counts_shape, torch.int32
        )
        work = manager.get_persistent(
            (*self._key, "work"), self._work_shape, torch.int32
        )
        workspace = manager.get_simultaneous(((self._workspace_bytes,), torch.int8))[0]
        return counts, work, workspace

    def _core(self, heads: int, qlen: int, causal: bool) -> "KvBalancedMla":
        key = (heads, qlen, causal)
        if key not in _SHAPES:
            shape = f"heads={heads} q={qlen} causal={causal}"
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    f"KIMI_K3_KV_BALANCED_MLA: first decode with {shape} inside CUDA "
                    "graph capture; the eager warmup must run every decode shape first"
                )
            from .core import KvBalancedMla, supports

            device = torch.device(
                current_platform.device_type, torch.accelerator.current_device_index()
            )
            if not supports(heads, qlen, device):
                raise ValueError(
                    f"KIMI_K3_KV_BALANCED_MLA can't decode {shape}: the kernel needs "
                    "more than 64 query rows per request (heads x query length)"
                )
            _SHAPES[key] = KvBalancedMla(heads, qlen, self.max_requests, causal, device)
            logger.info(
                "Kimi K3 KV-balanced MLA decode: %s max_requests=%d",
                shape,
                self.max_requests,
            )
        return _SHAPES[key]

    def __call__(
        self,
        impl: "MLACommonImpl",
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        layer: "AttentionLayer",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``impl``'s ``forward_mqa`` on the kernel."""
        if isinstance(q, tuple) or q.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "KIMI_K3_KV_BALANCED_MLA needs the packed FP8 query, got "
                + ("a (q_nope, q_pe) tuple" if isinstance(q, tuple) else str(q.dtype))
            )
        decode = attn_metadata.decode
        requests = attn_metadata.num_decodes
        tokens = attn_metadata.num_decode_tokens
        assert decode is not None and requests > 0
        unsupported = [
            reason
            for reason, failed in (
                (
                    f"{requests} requests (at most {self.max_requests})",
                    requests > self.max_requests,
                ),
                (
                    f"{tokens} tokens over {requests} requests (uniform, at most "
                    f"{self.max_q} each)",
                    tokens % requests or tokens // requests > self.max_q,
                ),
                (
                    f"a query of shape {tuple(q.shape)} (expected "
                    f"({tokens}, {self.heads}, {LATENT + ROPE}), contiguous)",
                    q.shape != (tokens, self.heads, LATENT + ROPE)
                    or not q.is_contiguous(),
                ),
            )
            if failed
        ]
        if unsupported:
            raise ValueError(
                "KIMI_K3_KV_BALANCED_MLA can't decode " + "; ".join(unsupported)
            )
        qlen = tokens // requests
        causal = bool(attn_metadata.causal)
        core = self._core(self.heads, qlen, causal)
        # The workspace manager raises for a micro-batch lane whose buffers were not
        # reserved before it was locked.
        counts, work, workspace = self._buffers()

        # One plan per forward, made by its first MLA layer.
        context, lens = get_forward_context(), decode.seq_lens.data_ptr()
        held = _PLANNED.get(work.data_ptr())
        if (
            held is None
            or held[0] is not context
            or held[1] != lens
            or held[2] is not core
        ):
            core.plan(decode.seq_lens, counts, work)
            _PLANNED[work.data_ptr()] = (context, lens, core)

        dcp = impl.dcp_world_size
        # Under DCP the causal mask needs the global lengths.
        assert dcp == 1 or decode.dcp_tot_seq_lens is not None
        lse = (
            torch.empty(
                (requests, qlen, self.heads), dtype=torch.float32, device=q.device
            )
            if impl.need_to_return_lse_for_decode
            else None
        )
        # The FP8 query and cache scales; V is stored as V / k_scale.
        q_scale = layer._q_scale_float
        k_scale = layer._k_scale_float
        out = core.decode(
            q,
            kv_c_and_k_pe_cache,
            decode.block_table,
            decode.seq_lens,
            impl.scale * q_scale * k_scale,
            k_scale,
            counts,
            work,
            workspace,
            lse=lse,
            causal_lens=decode.dcp_tot_seq_lens if dcp > 1 else None,
            cp=(
                (dcp, impl.dcp_rank, self.parallel_config.cp_kv_cache_interleave_size)
                if dcp > 1
                else (1, 0, 1)
            ),
        )
        return out, (lse.view(-1, self.heads) if lse is not None else None)
