# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlare Qwen3 draft model for speculative decoding.

DFlare ("flares out" DFlash's narrow conditioning bottleneck) extends the
DFlash Qwen3 drafter with two structural changes:

1. Separate context/noise K,V projections in each attention layer:
   - draft (noise) tokens still use the fused ``qkv_proj``'s K/V (== DFlash's
     ``k_proj`` / ``v_proj``);
   - target-hidden context tokens use dedicated ``kv_proj_target`` projections.
   DFlash shares one K/V projection for both sources.

2. Learnable per-draft-layer fusion of the T captured target layers:
   - DFlash collapses the T layers into ONE context tensor via a shared
     ``fc: Linear(T*D, D)`` — every draft layer sees the same context.
   - DFlare keeps the T layers un-collapsed and learns a
     ``layer_fusion_weights`` matrix ``[num_draft_layers, T]``. Each draft
     layer attends to its OWN softmax-weighted combination of the T target
     layers, giving every layer a distinct input.

The draft-side attention (query tokens against the pre-populated KV cache)
is otherwise identical to DFlash, so we can reuse the DFlash decoder layer
and the entire spec-decode runtime plumbing. The DFlare-specific work lives
in ``combine_hidden_states`` (no collapse) and ``precompute_and_store_context_kv``
(per-layer fusion + per-layer target K/V projection).
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight


class DFlareQwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash decoder layer + dedicated target-hidden K/V projections.

    ``k_proj_target`` / ``v_proj_target`` are used ONLY during context-KV
    precomputation (see ``DFlareQwen3Model.precompute_and_store_context_kv``);
    the draft-side forward pass still consumes ``qkv_proj``'s K/V for the noise
    tokens, so no changes to ``DFlashQwen3Attention.forward`` are required.

    We use two independent ``ColumnParallelLinear`` layers (one for K, one for
    V) rather than a single fused ``QKVParallelLinear`` with ``total_num_heads=0``.
    The fused approach turned out to leave the target-KV weight all-zero at
    inference (the QKV v2 loader's shard bookkeeping is unreliable with a
    zero-Q partition), so we now match the checkpoint layout directly: each
    layer stores its own ``k_proj_target.weight`` / ``v_proj_target.weight``
    of shape ``[num_kv_heads * head_dim, hidden]``.

    TP requirement: ``num_kv_heads`` must be divisible by ``tp_size`` because
    each ``ColumnParallelLinear`` shards along its output dim, i.e. along the
    concatenated K/V-head dimension. When ``tp_size > num_kv_heads`` a plain
    ColumnParallelLinear would split a single head across ranks and break the
    per-head layout the attention cache expects; ``DFlareQwen3Model.__init__``
    asserts this precondition.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        # Same bias semantics as qkv_proj (DFlash uses o_proj/qkv bias together
        # via ``attention_bias``); target K/V follow the same convention.
        attention_bias = getattr(config, "attention_bias", False)
        kv_output_size = config.num_key_value_heads * head_dim
        # NOTE: the target K/V projections are attached to ``self_attn`` (not
        # to the decoder layer directly) so the parameter path matches the
        # DFlare checkpoint layout, e.g.
        # ``layers.<i>.self_attn.k_proj_target.weight``. AutoWeightsLoader
        # dispatches by dotted path, so a mismatch here silently drops the
        # weight and leaves ``self.self_attn.k_proj_target.weight`` at its
        # uninitialized ``torch.empty`` contents (NaN / all-zero).
        self.self_attn.k_proj_target = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=kv_output_size,
            bias=attention_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.k_proj_target",
        )
        self.self_attn.v_proj_target = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=kv_output_size,
            bias=attention_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.v_proj_target",
        )


@support_torch_compile
class DFlareQwen3Model(DFlashQwen3Model):
    """DFlare draft-model backbone.

    Overrides vs. ``DFlashQwen3Model``:
      * Decoder layers are ``DFlareQwen3DecoderLayer`` (adds ``kv_proj_target``).
      * ``layer_fusion_weights[num_layers, num_target_layers]`` replaces the
        DFlash ``fc`` collapse; ``combine_hidden_states`` no longer projects to
        ``hidden_size`` and instead returns the raw concatenated ``T*D`` tensor.
      * ``precompute_and_store_context_kv`` fuses target layers per draft layer
        and uses the per-layer ``kv_proj_target`` for context K/V.
      * ``load_weights`` extends the stacked-params mapping to route
        ``k_proj_target`` / ``v_proj_target`` into ``kv_proj_target``.
    """

    # Route the parent's layer-construction hook to the DFlare layer so that
    # ``super().__init__`` builds ``self.layers`` with our subclass directly.
    # Re-building ``self.layers`` after ``super().__init__`` would double-
    # register each inner ``Attention``'s prefix and trigger
    # ``ValueError: Duplicate layer name: model.layers.<i>.self_attn.attn``.
    decoder_layer_class: type = DFlareQwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )

        # DFlare's target K/V projections are plain ``ColumnParallelLinear`` and
        # shard along the output dim (the concatenated KV-head dim). That
        # requires ``num_key_value_heads`` to be divisible by TP so we don't
        # split a single head across ranks. DFlash fused ``qkv_proj`` handles
        # tp_size > num_kv_heads via KV-head replication; DFlare does not.
        tp_size = get_tensor_model_parallel_world_size()
        num_kv_heads = self.config.num_key_value_heads
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                "DFlare requires num_key_value_heads "
                f"({num_kv_heads}) to be divisible by tensor_parallel_size "
                f"({tp_size}); its target K/V projections shard along the "
                "output dim and cannot replicate KV heads across ranks."
            )

        # DFlare fuses per draft layer, so it does not need DFlash's ``fc``
        # (which collapses T target layers into one shared context). Drop it so
        # its parameters are neither loaded nor optimized. ``combine_hidden_states``
        # becomes a passthrough of the raw ``[num_tokens, T*D]`` concatenation.
        if hasattr(self, "fc"):
            del self.fc
        # The aux-hidden-state flag still gates whether the speculator hands us
        # a concatenation of T layers vs. a single last-layer hidden. DFlare
        # always consumes T layers, so keep it on.
        self.use_aux_hidden_state = True

        # Resolve the number of target layers whose hidden states we consume.
        # Prefer the DFlash-style ``target_layer_ids`` list; fall back to the
        # generic ``eagle_aux_hidden_state_layer_ids`` set by the speculators
        # loader. DFlare checkpoints may store the same fields under
        # ``dflare_config`` (torchspec export) or expose ``target_layer_ids``
        # at the top level, so probe all three locations.
        drafter_config = {}
        drafter_config.update(getattr(self.config, "dflash_config", None) or {})
        drafter_config.update(getattr(self.config, "dflare_config", None) or {})
        target_layer_ids = drafter_config.get(
            "target_layer_ids",
            getattr(
                self.config,
                "target_layer_ids",
                getattr(self.config, "eagle_aux_hidden_state_layer_ids", None),
            ),
        )
        if not target_layer_ids:
            raise ValueError(
                "DFlare requires the set of target layer ids to be configured "
                "via dflash_config.target_layer_ids, dflare_config.target_layer_ids, "
                "config.target_layer_ids, or eagle_aux_hidden_state_layer_ids."
            )
        self.num_target_layers = len(target_layer_ids)
        self.num_draft_layers = self.config.num_hidden_layers

        # Learnable per-draft-layer fusion logits. Always loaded from the
        # checkpoint (no default init: the DFlare training pipeline ships
        # trained values, and running inference against random weights is
        # never the intended path).
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(
                self.num_draft_layers,
                self.num_target_layers,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    # Overrides for context KV precomputation
    # ------------------------------------------------------------------

    def _build_context_kv_buffers(self, layers_attn, has_bias: bool) -> None:
        """Stack per-layer target-K/V projection weights for fused GEMMs.

        Unlike DFlash — which pulls the K/V slice out of the shared ``qkv_proj``
        — DFlare has dedicated per-layer ``k_proj_target`` / ``v_proj_target``
        linear layers. We interleave their weights (K, then V, per layer) along
        the output dim so ``_project_context_kv`` can reshape to
        ``[L, 2, num_kv_heads, head_dim]`` exactly like DFlash does.
        """
        self._hidden_norm_weight = self.hidden_norm.weight.data

        # Interleave K/V per layer: [k_l0, v_l0, k_l1, v_l1, ...]. Each block
        # has shape ``[kv_size, hidden]`` after TP sharding of the K/V head
        # dim, giving a final fused tensor of ``[L * 2 * kv_size, hidden]``
        # whose ``.view(L, 2, nkv, hd, ...)`` layout matches DFlash.
        target_kv_weights = []
        target_kv_biases = []
        for layer in self.layers:
            target_kv_weights.append(layer.self_attn.k_proj_target.weight)
            target_kv_weights.append(layer.self_attn.v_proj_target.weight)
            if has_bias:
                target_kv_biases.append(layer.self_attn.k_proj_target.bias)
                target_kv_biases.append(layer.self_attn.v_proj_target.bias)
        self._fused_kv_weight = torch.cat(target_kv_weights, dim=0)
        if has_bias:
            self._fused_kv_bias: torch.Tensor | None = torch.cat(
                target_kv_biases, dim=0
            )
        else:
            self._fused_kv_bias = None

        # Per-layer K-norm still lives on the attention module; keep it as a
        # stacked tensor for the grouped-kernel RMSNorm across layers.
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse the T target layers per draft layer, then run a single fused
        target-KV BMM across all L layers.

        Args:
            context_states: ``[num_ctx, T * D]`` — the raw concatenation of the
                T captured target hidden states, as returned by
                ``combine_hidden_states`` (which is a no-op passthrough for
                DFlare).
        """
        L, T = self.num_draft_layers, self.num_target_layers
        D = self.config.hidden_size

        # 1. Un-collapse the T target layers so we can fuse them per draft
        #    layer with distinct softmax weights.
        # [num_ctx, T*D] -> [num_ctx, T, D]
        stacked = context_states.view(num_ctx, T, D)

        # 2. Per-draft-layer fusion. layer_fusion_weights is [L, T]; softmax
        #    over the T dim gives a probability distribution per draft layer.
        fusion_probs = F.softmax(
            self.layer_fusion_weights.to(dtype=context_states.dtype), dim=-1
        )  # [L, T]
        # Weighted combine: [L, T] x [num_ctx, T, D] -> [L, num_ctx, D].
        fused = torch.einsum("lt,ntd->lnd", fusion_probs, stacked)

        # 3. RMSNorm before the per-layer target-K/V projection. DFlare shares
        #    ``hidden_norm`` (aliased to the training-side ``context_norm``)
        #    across all draft layers, matching the reference implementation.
        fused_flat = fused.reshape(L * num_ctx, D)
        normed = torch.empty_like(fused_flat)
        ops.rms_norm(
            normed,
            fused_flat,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        normed = normed.view(L, num_ctx, D)

        # 4. Per-layer K/V projection via one BMM across the L dim. Because the
        #    target-KV projection is *distinct per draft layer*, we cannot
        #    flatten across L into one big linear.
        kv_size_per_partition = 2 * num_kv_heads * head_dim
        w_stacked = self._fused_kv_weight.view(L, kv_size_per_partition, D)
        # [L, N, D] @ [L, D, 2*kv] -> [L, N, 2*kv]
        all_kv_flat = torch.bmm(normed, w_stacked.transpose(1, 2))
        if self._fused_kv_bias is not None:
            b_stacked = self._fused_kv_bias.view(L, 1, kv_size_per_partition)
            all_kv_flat = all_kv_flat + b_stacked

        # 5. Split K / V and lay out as [L, num_ctx, nkv, hd] contiguous —
        #    the exact layout DFlash's normalize / RoPE / cache-write path
        #    expects.
        all_kv = all_kv_flat.view(L, num_ctx, 2, num_kv_heads, head_dim)
        all_kv = all_kv.permute(2, 0, 1, 3, 4).contiguous()
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd]
        all_v = all_kv[1]
        return all_k, all_v

    # ------------------------------------------------------------------
    # combine_hidden_states override
    # ------------------------------------------------------------------

    def combine_hidden_states(
        self, aux_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """DFlare skips the DFlash ``fc`` collapse and returns the raw
        ``[num_tokens, T*D]`` concatenation. Per-layer fusion happens later,
        inside ``precompute_and_store_context_kv``.
        """
        return aux_hidden_states

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Same as DFlash's loader but with an early-return for the DFlare
        per-layer ``k_proj_target`` / ``v_proj_target`` weights.

        Without this early-return, the stacked-params rule ``(".qkv_proj",
        ".k_proj", "k")`` would false-match ``.k_proj_target`` (substring hit),
        rewrite the name to ``qkv_proj_target`` and blow up with a KeyError
        against ``params_dict``. ``layer_fusion_weights`` and the ordinary
        ``k_proj_target`` / ``v_proj_target`` module weights are then loaded
        via the default per-parameter weight loader.
        """
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            if "attention_sink_bias" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                heads_per_rank = loaded_weight.shape[0] // tp_size
                head_start = tp_rank * heads_per_rank
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            # DFlare-specific target projections must bypass the ".k_proj" /
            # ".v_proj" stacked-mapping substring match (``.k_proj`` is a
            # substring of ``.k_proj_target``). They are plain ColumnParallel
            # linears, so their own weight_loader knows how to shard them.
            if ".k_proj_target" in name or ".v_proj_target" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    # Silently ignore extra checkpoint tensors that inference
                    # does not consume (e.g. legacy ``fc`` / ``context_proj``
                    # left over from a copy-converted DFlash checkpoint).
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


class DFlareQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level wrapper mirroring ``DFlashQwen3ForCausalLM`` but using the
    DFlare backbone.

    Overrides:
      * builds a ``DFlareQwen3Model`` instead of ``DFlashQwen3Model``;
      * ``load_weights`` renames a few DFlare-specific weight paths
        (``context_norm`` -> ``hidden_norm``, drop ``context_proj`` / ``fc``)
        before delegating to the standard AutoWeightsLoader.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlareQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DFlare does not project the aux hidden states — the concatenation is
        # the ``context_feature`` and per-layer fusion happens inside
        # ``precompute_and_store_context_kv``. ``combine_hidden_states`` on the
        # inner model is a passthrough; we mirror it here for the speculator.
        return self.model.combine_hidden_states(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlare embeds masked slots via mask_token_id (optionally "
                "overridden by a mask_embedding.pt file); it should not ship "
                "a mask_hidden weight."
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            # Aliasing rules for training-side names that differ from vLLM:
            #   - training uses ``context_norm`` for the pre-target-KV RMSNorm;
            #     vLLM reuses ``hidden_norm`` for the same role.
            #   - training uses ``final_norm`` for the output RMSNorm; vLLM
            #     calls it ``norm`` (see DFlashQwen3Model).
            if name == "model.context_norm.weight":
                name = "model.hidden_norm.weight"
            if name == "model.final_norm.weight":
                name = "model.norm.weight"
            # DFlare drops DFlash's ``context_proj`` / ``fc`` (the T-layer
            # collapse) in favor of per-layer fusion — any leftover checkpoint
            # entry is not needed at inference.
            if "context_proj" in name or name.startswith("model.fc."):
                continue
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs: list[str] = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        # DFlare has no aux-linear ``fc`` (per-layer fusion replaces it).
        skip_substrs.append("fc.")
        # Not currently wired for inference; skip its weights if shipped.
        skip_substrs.append("mask_embedding")

        loader = AutoWeightsLoader(self, skip_prefixes=None, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())

        # Buffers used by precompute_and_store_context_kv have to be built
        # AFTER weights are loaded (they alias the loaded weight tensors).
        self.model._build_fused_kv_buffers()


__all__ = [
    "DFlareQwen3DecoderLayer",
    "DFlareQwen3Model",
    "DFlareQwen3ForCausalLM",
]
