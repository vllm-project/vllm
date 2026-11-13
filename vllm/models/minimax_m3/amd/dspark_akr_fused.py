# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The rest of the amd_kernels ``draft`` route's DSpark forward at TP4 on the library
block ``amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4``: the input fc split by N
(eager), the layer stack around the draft attention and the context K/V chain."""

import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.rocm_akr import is_akr_layer
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

logger = init_logger(__name__)


_H = 6144  # hidden size of the kernels
_SWIGLU_FORM = "bf16_chain"  # the uncompiled SiluAndMulWithClamp's aten chain


class DraftFused:
    """The library block of the draft's layers, built collectively on the 4 TP ranks
    once the draft shares the target's embedding."""

    def __init__(self, model: torch.nn.Module, dense: list) -> None:
        from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import (
            MAX_ROWS,
            DenseDecoderRanks4,
        )

        from vllm.distributed.parallel_state import get_tp_group

        inner = model.model
        tp = get_tp_group()
        self.layers = list(inner.layers)
        self.dense = dense
        self.eps = inner.norm.variance_epsilon
        act = self.layers[0].mlp.act_fn
        self.swiglu = (act.alpha, act.beta, act.swiglu_limit)
        # the block keeps this rank's 1536 fc rows; the replicated weight is released
        fc = inner.fc
        w = fc.weight.data
        self.blk = DenseDecoderRanks4(
            tp.rank_in_group,
            tp.cpu_group,
            tp.device,
            fc_weight=w,
            embedding=inner.embed_tokens.weight,
            max_rows=MAX_ROWS,
        )
        fc.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
        del w
        self.ctx: bool | None = None  # the context K/V chain, decided once caches bind
        for layer in self.layers:
            # TP partials (the next tail reduces them); o_proj reads the attention's
            # MXFP8 epilogue
            layer.self_attn.o_proj.reduce_results = False
            layer.mlp.down_proj.reduce_results = False
            layer.self_attn._akr_mxfp8_out = layer.self_attn._akr_query
        torch.accelerator.empty_cache()

    def fc_into(
        self, taps: list[torch.Tensor], num_tokens: int, out: torch.Tensor
    ) -> None:
        """The speculator's ``akr_fc_into`` hook: out[:num_tokens] =
        fc(cat(taps)[:num_tokens]), the 5 taps read in place (eager, every rank)."""
        self.blk.fc(taps, num_tokens, out)

    def forward(
        self, inner, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """DFlashQwen3Model.forward (embedding, the layers, the final norm)."""
        m = input_ids.numel()

        def attn(i: int, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            act = QuantizedActivation(
                q, s, torch.bfloat16, torch.Size((m, _H)), kMxfp8Dynamic
            )
            return self.layers[i].self_attn(positions, act)  # o_proj's TP partial

        return self.blk.forward(
            input_ids,
            self.dense,
            attn,
            inner.norm.weight,
            self.eps,
            self.swiglu,
            form=_SWIGLU_FORM,
        )

    def store_context_kv(
        self,
        inner,
        states: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> bool:
        """``precompute_and_store_context_kv`` on the library's norm + GEMM + insert
        chain; False where the stock path runs."""
        from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import (
            context_kv_blocker,
        )

        layers = inner._attn_layers
        caches = [a.kv_cache for a in layers]
        k_scales = [a._k_scale for a in layers]
        v_scales = [a._v_scale for a in layers]
        proj = inner._context_kv_proj
        if self.ctx is None:
            if any(c.numel() == 0 for c in caches):
                return False  # not bound yet: decide later
            why: str | None
            if inner.layers[0].self_attn.v_scale is not None:
                why = "a v_scale"
            elif not all(str(a.kv_cache_dtype).startswith("fp8") for a in layers):
                why = "a KV cache not fp8"
            else:
                why = context_kv_blocker(
                    is_akr_layer(proj),
                    proj.weight,
                    inner.layers[0].self_attn.rotary_emb.rotary_dim,
                    inner._rope_is_neox,
                    inner._rope_cos_sin_cache,
                    caches,
                    k_scales,
                    v_scales,
                )
            self.ctx = why is None
            if why is not None:
                logger.warning(
                    "amd_kernels draft: the context K/V stays stock (%s)", why
                )
        if not self.ctx or states.dtype != torch.bfloat16:
            return False
        self.blk.store_context_kv(
            states,
            inner.hidden_norm.weight,
            inner.hidden_norm.variance_epsilon,
            proj.weight,
            proj.weight_scale,
            positions,
            inner._k_norm_weights,
            inner._rms_norm_eps,
            inner._rope_cos_sin_cache,
            caches,
            slot_mapping,
            k_scales,
            v_scales,
        )
        return True

    def capture(self):
        """Around the draft's graph capture (registers the tails' peer inputs)."""
        return self.blk.capture()


def setup(model: torch.nn.Module, max_query_rows: int) -> DraftFused | None:
    """The fused forward of a loaded draft (every TP rank, before graph capture), or
    None where it stays off."""
    from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import (
        DenseLayer,
        stack_blocker,
    )

    from vllm.distributed import get_tensor_model_parallel_world_size

    inner = model.model
    dense: list = []
    why: str | None
    if inner.has_separate_mask_embedding or not inner.use_aux_hidden_state:
        why = "a separate mask embedding or no aux fc"
    else:
        on_library: list[bool] = []
        for x in inner.layers:
            a, mlp = x.self_attn, x.mlp
            linears = (a.qkv_proj, a.o_proj, mlp.gate_up_proj, mlp.down_proj)
            on_library += [is_akr_layer(lin) for lin in linears]
        if all(on_library):  # the weights are in the library's layout
            dense = [
                DenseLayer(
                    x.input_layernorm.weight,
                    tuple(x.self_attn.qkv_proj.weight.shape),
                    x.post_attention_layernorm.weight,
                    x.mlp.gate_up_proj.weight,
                    x.mlp.gate_up_proj.weight_scale,
                    x.mlp.down_proj.weight,
                    x.mlp.down_proj.weight_scale,
                )
                for x in inner.layers
            ]
        why = stack_blocker(
            get_tensor_model_parallel_world_size(),
            max_query_rows,
            on_library,
            inner.fc.weight,
            inner.embed_tokens.weight,
            dense,
        )
    if why is not None:
        logger.warning("amd_kernels draft: the fused draft stays off (%s)", why)
        return None
    t0 = time.perf_counter()
    fused = DraftFused(model, dense)
    logger.info(
        "amd_kernels draft: the input fc, embedding + norm, all-reduce + norm tails, "
        "MLP blocks and context K/V run amd_kernels (set-up and warm-up %.1f s)",
        time.perf_counter() - t0,
    )
    return fused
