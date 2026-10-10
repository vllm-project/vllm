# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routes the decoder layers of eligible GLM-5 decode steps to the MonoKernel.

Each layer launch all-reduces inside the kernel, so whether a step takes the
kernel must be decided identically on every TP rank: it depends only on the
token count and the attention metadata (``spec.step_reason``), decided once per
forward at the first layer.

A kernel layer takes and returns vLLM's ``(hidden_states, residual)``: it
returns a zero partial sum and the full residual stream, so the next layer (or
the final norm) adds nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.platforms import current_platform

from . import spec
from .weights import _computes_indices, _layer_weights

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.deepseek_v32.amd.model import (
        DeepseekV32DecoderLayer,
        DeepseekV32Model,
    )
    from vllm.models.deepseek_v32.amd.ops.mono import Glm5MonoKernel

logger = init_logger(__name__)


def refuse_config(vllm_config: VllmConfig) -> None:
    """Config-stage refusal, before weights load: raises with every reason."""
    on_gfx950 = False
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950 as _on_gfx950

        on_gfx950 = _on_gfx950()
    reasons = spec.refuse_config(vllm_config, on_gfx950)
    if reasons:
        raise ValueError(
            "VLLM_ROCM_MONO_DECODE cannot run GLM-5 here: " + "; ".join(reasons) + "."
        )


def forward_layer(
    mono: GlmMono | None,
    idx: int,
    layer: DeepseekV32DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mono is None:
        return layer(positions, hidden_states, residual)
    return mono.forward_layer(idx, layer, positions, hidden_states, residual)


class GlmMono:
    """The MonoKernel path of one rank's ``DeepseekV32Model``.

    Weights are viewed or repacked once after loading. The per-width kernels are
    built on the first kernel step, which runs eagerly before CUDA graph capture,
    because their index-cache constants come from the attention metadata.
    """

    def __init__(self, model: DeepseekV32Model, vllm_config: VllmConfig) -> None:
        self._model = model
        layers = list(model.layers)
        self._layers = layers
        self._first, self._last = 0, len(layers) - 1
        self._num_dense = next(
            (
                i
                for i, layer in enumerate(layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            ),
            len(layers),
        )
        self._layer_name = layers[min(self._num_dense, self._last)].self_attn.layer_name
        self._max_model_len = vllm_config.model_config.max_model_len
        self._ops: dict[int, list[Glm5MonoKernel]] = {}
        self._runtimes: dict[int, list[Glm5MonoKernel]] = {}
        self._launch_index: list[int] = []
        self._cache_key: tuple | None = None
        self._index_options: dict = {}
        self.active = False
        self._rows = 0
        self._step: dict = {}

        self._refuse_model()
        from vllm.models.deepseek_v32.amd.ops.mono import (
            AttentionWeight,
            glm5_tp_config,
            prepare_glm5_weights,
        )

        tp = get_tensor_model_parallel_world_size()
        self._tp, self._rank = tp, get_tensor_model_parallel_rank()
        self._cfg = glm5_tp_config(tp)
        self._fused = [_computes_indices(layer) for layer in layers]
        dense = [i < self._num_dense for i in range(len(layers))]
        weights, reasons = [], []
        for i, (layer, f, d) in enumerate(zip(layers, self._fused, dense)):
            try:
                weights.append(
                    _layer_weights(
                        layer, self._cfg, self._rank, tp, with_indexer=f, dense=d
                    )
                )
            except ValueError as error:
                reasons.append(f"layer {i}: {error}")
        if reasons:
            raise ValueError(
                "VLLM_ROCM_MONO_DECODE cannot run this model: " + "; ".join(reasons)
            )
        prepared = [
            prepare_glm5_weights(w, AttentionWeight.FP8_BLOCK128) for w in weights
        ]
        # Drop the unpacked attention copies; every op reads the packed ones.
        for w, p in zip(weights, prepared):
            for name in ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o"):
                w.t[name] = p[name]
        self._weights, self._prepared = weights, prepared
        self._dense = dense

        cos_sin = layers[0].self_attn.rotary_emb.cos_sin_cache
        half = cos_sin.shape[-1] // 2
        self._cos = cos_sin[:, :half].to(torch.bfloat16).contiguous()
        self._sin = cos_sin[:, half:].to(torch.bfloat16).contiguous()
        self._cur_pos = torch.zeros(1, dtype=torch.int32, device=cos_sin.device)
        self._zeros = torch.zeros(
            spec.width_for(max(spec.STEP_TOKENS)),
            model.config.hidden_size,
            dtype=torch.bfloat16,
            device=cos_sin.device,
        )
        logger.info_once(
            "GLM-5 MonoKernel: %d layers (%d dense, %d with the fused indexer), "
            "decode steps of %s tokens.",
            len(layers),
            self._num_dense,
            sum(self._fused),
            "/".join(map(str, spec.STEP_TOKENS)),
        )

    def _refuse_model(self) -> None:
        """Model-stage refusal, after weights load: raises with every reason."""
        model, layers = self._model, self._layers
        reasons = []
        if not model.is_fused_shared_expert_enabled:
            reasons.append(
                "needs VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 (shared expert "
                "in the last expert slot)"
            )
        if not all(
            isinstance(layer.mlp, DeepseekV2MoE) for layer in layers[self._num_dense :]
        ):
            reasons.append("needs every layer after the leading dense ones to be MoE")
        attn = next(
            (layer.self_attn for layer in layers if _computes_indices(layer)), None
        )
        if attn is not None and not torch.equal(
            attn.rotary_emb.cos_sin_cache, attn.indexer_rope_emb.cos_sin_cache
        ):
            reasons.append("needs the indexer and MLA RoPE tables to match")
        if reasons:
            raise ValueError(
                "VLLM_ROCM_MONO_DECODE cannot run this model: " + "; ".join(reasons)
            )

    def forward_layer(
        self,
        idx: int,
        layer: DeepseekV32DecoderLayer,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if idx == self._first:
            self.active = self._begin_step(positions, hidden_states, residual)
        if not self.active:
            return layer(positions, hidden_states, residual)
        state = hidden_states if residual is None else residual
        state = self._launch(idx, layer, state)
        if idx == self._last:
            for runtime in self._runtimes[spec.width_for(self._rows)]:
                runtime.advance_step()
        return self._zeros[: state.shape[0]], state

    def final_norm(self, norm: RMSNorm, residual: torch.Tensor) -> torch.Tensor:
        return norm(residual[: self._rows])

    def _begin_step(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> bool:
        if residual is not None or torch.compiler.is_compiling():
            return False
        n = hidden_states.shape[0]
        ctx = get_forward_context()
        if spec.step_reason(n, ctx.attn_metadata, self._layer_name) is not None:
            return False
        md = ctx.attn_metadata[self._layer_name]
        if not self._ops:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "VLLM_ROCM_MONO_DECODE was first reached inside CUDA graph "
                    "capture, before an eager decode step."
                )
            self._build(md)
        self._check_caches(md)
        slots = ctx.slot_mapping[self._layer_name][:n]
        indptr = md.paged_kv_indptr[: n + 1]
        req_ids = md.req_id_per_token[:n]
        if n == 1:
            positions = torch.cat([positions, positions])
            slots = torch.cat([slots, torch.full_like(slots, -1)])
            indptr = torch.cat([indptr, indptr[1:]])
            req_ids = torch.cat([req_ids, req_ids])
        self._rows = n
        self._step = dict(
            md=md, positions=positions, slots=slots, indptr=indptr, req_ids=req_ids
        )
        return True

    def _launch(
        self, idx: int, layer: DeepseekV32DecoderLayer, state: torch.Tensor
    ) -> torch.Tensor:
        st = self._step
        if state.shape[0] == 1:
            state = torch.cat([state, torch.zeros_like(state)])
        op = self._ops[spec.width_for(self._rows)][idx]
        md = st["md"]
        attn = layer.self_attn
        cache = attn.kv_cache.view(current_platform.fp8_dtype()).view(
            -1, attn.kv_cache.shape[-1]
        )
        if op.with_indexer:
            return op.forward(
                state,
                self._cur_pos,
                cache,
                cache,
                None,
                self._cos,
                self._sin,
                layer=self._launch_index[idx],
                advance=False,
                positions=st["positions"],
                slot_mapping=st["slots"],
                sparse_kv_indptr=st["indptr"],
                index_cache=attn.indexer.k_cache.kv_cache,
                block_table=md.block_table,
                req_ids=st["req_ids"],
                out_indices=md.paged_kv_indices,
            )
        return op.forward(
            state,
            self._cur_pos,
            cache,
            cache,
            md.paged_kv_indices,
            self._cos,
            self._sin,
            layer=self._launch_index[idx],
            advance=False,
            positions=st["positions"],
            slot_mapping=st["slots"],
            sparse_kv_indptr=st["indptr"],
        )

    def _build(self, md) -> None:
        from vllm.models.deepseek_v32.amd.ops.mono import (
            AttentionWeight,
            Glm5MonoKernel,
            KvCacheLayout,
        )

        layers, fused = self._layers, self._fused
        self._index_options = self._index_options_from(md) if any(fused) else {}
        group = get_tp_group().cpu_group
        launches = {f: fused.count(f) for f in (False, True)}
        self._launch_index = [fused[:i].count(f) for i, f in enumerate(fused)]
        for chunk in sorted({spec.width_for(n) for n in spec.STEP_TOKENS}):
            runtimes: dict[bool, Glm5MonoKernel | None] = {False: None, True: None}
            ops = []
            for w, p, f, d in zip(self._weights, self._prepared, fused, self._dense):
                op = Glm5MonoKernel(
                    w,
                    chunk,
                    rank=self._rank,
                    npes=self._tp,
                    group=group,
                    topk=spec.SPARSE_TOPK,
                    launches_per_step=launches[f],
                    with_indexer=f,
                    attention_weight=AttentionWeight.FP8_BLOCK128,
                    kv_cache_layout=KvCacheLayout.ATOM,
                    kv_cache_dtype="fp8",
                    prepared_weights=p,
                    runtime=runtimes[f],
                    native_fp4_mfma=True,
                    dense_experts=w.physical_experts if d else 0,
                    **(self._index_options if f else {}),
                )
                runtimes[f] = runtimes[f] or op
                ops.append(op)
            self._ops[chunk] = ops
            self._runtimes[chunk] = [r for r in runtimes.values() if r is not None]
        assert len(self._ops[spec.width_for(1)]) == len(layers)

    def _index_options_from(self, md) -> dict:
        attn = next(layer.self_attn for layer, f in zip(self._layers, self._fused) if f)
        cache = attn.indexer.k_cache.kv_cache
        block = md.block_size
        return dict(
            index_max_seq=(self._max_model_len + block - 1) // block * block,
            index_paged=True,
            index_block_size=cache.shape[1],
            index_block_bytes=cache.stride(0) * cache.element_size(),
            index_shuffled=attn.indexer.k_cache.uses_shuffled_layout,
            block_table_stride=md.block_table.stride(0),
            index_k_bf16=True,
        )

    def _check_caches(self, md) -> None:
        """Cache-stage check, re-run whenever a cache tensor moved (CUDA-graph
        memory profiling runs real steps on caches that are later replaced)."""
        attn = self._layers[self._first].self_attn
        key = (
            attn.kv_cache.data_ptr(),
            attn.kv_cache.numel(),
            md.block_table.stride(0),
        )
        if key == self._cache_key:
            return
        reasons = []
        if attn.kv_cache.shape[-1] != self._cfg.kv_lora + self._cfg.pe_dim:
            reasons.append(f"KV cache rows of {attn.kv_cache.shape[-1]} bytes")
        if self._index_options:
            opts = self._index_options
            if md.block_table.stride(0) != opts["block_table_stride"]:
                reasons.append(
                    f"block table stride {md.block_table.stride(0)}, built for "
                    f"{opts['block_table_stride']}"
                )
            for layer, f in zip(self._layers, self._fused):
                if not f:
                    continue
                cache = layer.self_attn.indexer.k_cache.kv_cache
                if (
                    cache.shape[1] != opts["index_block_size"]
                    or cache.stride(0) * cache.element_size()
                    != opts["index_block_bytes"]
                ):
                    reasons.append(
                        f"index cache {tuple(cache.shape)} strides {cache.stride()}"
                    )
                    break
        if reasons:
            raise ValueError(
                "VLLM_ROCM_MONO_DECODE caches changed layout: " + "; ".join(reasons)
            )
        self._cache_key = key
