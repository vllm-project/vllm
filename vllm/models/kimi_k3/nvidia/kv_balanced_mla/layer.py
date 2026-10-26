# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 MLA decode on the KV-balanced kernel, in place of the TOKENSPEED_MLA
backend's ``forward_mqa``.

Selected with ``additional_config={"mla_decode_backend": "kv_balanced"}``; the default
``"auto"`` keeps the attention backend's decode. It covers the target verify and the
DSpark draft, with or without DCP. Per forward, the first MLA layer of each decode
shape plans the split-KV work list from this rank's KV lengths, and every later layer
of that forward reuses it. Under DCP the kernel reads this rank's KV shard, masks
causally in global positions and returns the log2 LSE that the DCP combine merges,
like TokenSpeed.

The layers share one set of plan buffers per micro-batch lane, reserved with the
split-KV scratch (the decode workspace TokenSpeed also uses) when the layers are
built, so the KV cache is sized around them. A step the kernel does not run uses the
backend's decode: a shape TokenSpeed runs on its M64 tiles (64 query rows or fewer
per request; the kernel needs the 2-CTA M128 tiles), more requests than its partials
can address, a strided query, a micro-batch lane first seen after the workspace is
locked, or the first call for a shape inside CUDA graph capture (the eager warmup
runs first).
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import max_decode_query_len
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

    from .core import KvBalancedMla

logger = init_logger(__name__)

AUTO, KV_BALANCED = "auto", "kv_balanced"
LATENT, ROPE = 512, 64


def mla_decode_backend(vllm_config: VllmConfig) -> str:
    """The ``mla_decode_backend`` of ``additional_config``: "auto" or "kv_balanced"."""
    extra = vllm_config.additional_config
    backend = extra.get("mla_decode_backend", AUTO) if isinstance(extra, dict) else AUTO
    if backend not in (AUTO, KV_BALANCED):
        raise ValueError(
            f"mla_decode_backend must be '{AUTO}' or '{KV_BALANCED}', got {backend!r}"
        )
    return backend


# The kernel per (heads, q, causal): the target verify and the DSpark draft. None:
# unsupported.
_SHAPES: dict[tuple[int, int, bool], "KvBalancedMla | None"] = {}
# Per plan buffer (``work``): the (forward context, KV lengths, kernel) of the plan it
# holds. The context is held so that its identity can't be reused by a later forward.
_PLANNED: dict[int, tuple[object, int, "KvBalancedMla"]] = {}


class TokenspeedKvBalancedMqa:
    """``forward_mqa`` of one Kimi K3 MLA layer on the KV-balanced kernel, with the
    TOKENSPEED_MLA backend's ``forward_mqa`` for the steps the kernel doesn't run."""

    def __init__(self, layer: "MultiHeadLatentAttention"):
        from . import core

        config = get_current_vllm_config()
        # NIXL P/D resolves the KV interleave after the model is built: read it per
        # call.
        self.parallel_config = config.parallel_config
        dcp = config.parallel_config.decode_context_parallel_size
        # Under DCP the query is gathered over the ranks' heads.
        self.heads = layer.num_local_heads * dcp
        self.max_q = max_decode_query_len(config)
        device = torch.device(current_platform.device_type, torch.cuda.current_device())
        if not core.supports(self.heads, self.max_q, device):
            raise RuntimeError(
                "mla_decode_backend=kv_balanced needs more than 64 query rows per "
                f"request (heads x query length), got {self.heads} x {self.max_q}"
            )
        if config.attention_config.tokenspeed_mla_min_split_kv > 1:
            logger.warning_once(
                "Kimi K3 KV-balanced MLA decode plans its own KV splits; "
                "tokenspeed_mla_min_split_kv applies only to the backend decode"
            )
        max_seqs = config.scheduler_config.max_num_seqs
        self.max_requests = min(
            max_seqs, core.max_planned_requests(self.heads, self.max_q)
        )
        if self.max_requests < max_seqs:
            logger.warning_once(
                "Kimi K3 KV-balanced MLA decode: steps above %d requests use the "
                "backend decode",
                self.max_requests,
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

    def _core(self, heads: int, qlen: int, causal: bool) -> "KvBalancedMla | None":
        key = (heads, qlen, causal)
        if key not in _SHAPES:
            if torch.cuda.is_current_stream_capturing():
                logger.warning_once(
                    "Kimi K3 KV-balanced MLA decode: first call for heads=%d q=%d "
                    "causal=%s inside CUDA graph capture; using the backend decode",
                    heads,
                    qlen,
                    causal,
                )
                return None
            from .core import KvBalancedMla, supports

            device = torch.device(
                current_platform.device_type, torch.cuda.current_device()
            )
            core = (
                KvBalancedMla(heads, qlen, self.max_requests, causal, device)
                if supports(heads, qlen, device)
                else None
            )
            _SHAPES[key] = core
            logger.info(
                "Kimi K3 KV-balanced MLA decode: heads=%d q=%d causal=%s %s",
                heads,
                qlen,
                causal,
                f"max_requests={self.max_requests}"
                if core is not None
                else "unsupported, using the backend decode",
            )
        return _SHAPES[key]

    def __call__(
        self,
        layer: "MultiHeadLatentAttention",
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        attn_metadata: "MLACommonMetadata",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self._decode(layer, q, attn_metadata)
        if out is None:
            out = layer.impl.forward_mqa(  # type: ignore[attr-defined]
                q, layer._attn_read_kv_cache(), attn_metadata, layer
            )
        return out

    def _decode(
        self,
        layer: "MultiHeadLatentAttention",
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        attn_metadata: "MLACommonMetadata",
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """The kernel's decode, or None for a step it doesn't run."""
        if isinstance(q, tuple) or q.dtype != torch.float8_e4m3fn:
            return None
        decode = attn_metadata.decode
        requests = attn_metadata.num_decodes
        tokens = attn_metadata.num_decode_tokens
        if (
            decode is None
            or requests == 0
            or requests > self.max_requests
            or tokens % requests
            or tokens // requests > self.max_q
            or q.shape[0] != tokens
            or q.shape[1] != self.heads
            or q.shape[-1] != LATENT + ROPE
            or not q.is_contiguous()
        ):
            return None
        qlen = tokens // requests
        causal = bool(attn_metadata.causal)
        core = self._core(self.heads, qlen, causal)
        if core is None:
            return None
        try:
            counts, work, workspace = self._buffers()
        except AssertionError:
            # A micro-batch lane whose buffers were not reserved before the lock.
            logger.warning_once(
                "Kimi K3 KV-balanced MLA decode: no buffers for this micro-batch lane; "
                "using the backend decode"
            )
            return None

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

        impl = layer.impl
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
        q_scale = cast(float, layer._q_scale_float)
        k_scale = cast(float, layer._k_scale_float)
        out = core.decode(
            q,
            layer._attn_read_kv_cache(),
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


def use_tokenspeed_kv_balanced_mqa(layer: "MultiHeadLatentAttention") -> bool:
    """Whether ``layer`` decodes on the KV-balanced kernel: ``mla_decode_backend``
    is "kv_balanced". Raises if the layer can't run it."""
    if mla_decode_backend(get_current_vllm_config()) != KV_BALANCED:
        return False
    backend = layer.attn_backend.get_name()
    unsupported = [
        reason
        for reason, failed in (
            (
                f"the TOKENSPEED_MLA attention backend (got {backend})",
                backend != "TOKENSPEED_MLA",
            ),
            (
                "an SM100-family GPU",
                not current_platform.is_device_capability_family(100),
            ),
            (
                f"an FP8 E4M3 KV cache (got {layer.kv_cache_dtype})",
                layer.kv_cache_dtype not in ("fp8", "fp8_e4m3"),
            ),
            (
                f"kv_lora_rank {LATENT} and qk_rope_head_dim {ROPE}",
                (layer.kv_lora_rank, layer.qk_rope_head_dim) != (LATENT, ROPE),
            ),
        )
        if failed
    ]
    if unsupported:
        raise RuntimeError(
            "mla_decode_backend=kv_balanced requires " + "; ".join(unsupported)
        )
    return True
