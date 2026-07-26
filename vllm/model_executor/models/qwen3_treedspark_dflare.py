# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TreeFlash + DSpark + DFlare draft model for speculative decoding.

This module combines three orthogonal enhancements on top of the DFlash Qwen3
draft into a single inference-side class:

  1. **DFlare backbone** (from ``qwen3_dflare.py``): separate context/noise
     K/V projections per attention layer, plus a learnable per-draft-layer
     softmax fusion of the T target hidden states (replacing DFlash's single
     ``fc`` collapse).
  2. **DSpark Markov head** (from ``qwen3_dspark.py``): low-rank ``V x r`` /
     ``r x V`` sequential transition bias added to the base draft logits,
     sampled left-to-right by the speculator.
  3. **TreeFlash hidden-states correction** (new): a lightweight residual
     SwiGLU applied to the drafter's output hidden state, conditioned on the
     previous token's embedding, that refines ``h`` before it enters the LM
     head. Zero-initialized on the down-projection so the correction starts
     at zero and the model degenerates to DFlare + Markov at init.

The training-side reference lives in torchspec at
``torchspec/models/draft/{dspark,dflare}.py`` and the top-level
``TreeDflashDSparkDFlareDraftModel``. The checkpoint's config carries
``architectures=["Qwen3TreeDSparkDFlareModel"]``, ``model_arch="dflare"``,
``markov_rank``, ``enable_hidden_correction``, and the DFlash-family target
layer specification (``target_layer_ids`` / ``dflare_config``).

The speculator dispatch treats this as ``method="dspark"``; the sequential
Markov sampling loop in ``DSparkSpeculator._sample_sequential`` will invoke
``apply_hidden_correction`` (defined below) once per step, feeding the
previously sampled token's embedding to refine the hidden state before the
LM head.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .qwen3_dflare import DFlareQwen3Model
from .qwen3_dspark import (
    DSparkMarkovHead,
    HiddenStatesCorrection,
    Qwen3DSparkForCausalLM,
)
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class Qwen3TreeDSparkDFlareModel(DFlareQwen3Model):
    """DFlare backbone + DSpark Markov head + optional TreeFlash hidden correction."""

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
        config = self.config

        # ---- DSpark Markov head (optional) ----
        # Toggle: when the checkpoint does not provide ``markov_rank`` (or sets
        # it to <= 0), we skip constructing the Markov head entirely. The
        # speculator's per-step loop then falls through the ``has_markov()``
        # check and does not add a Markov bias.
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_rank = int(getattr(config, "markov_rank", 0) or 0)
        if markov_rank > 0:
            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                draft_vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
            )
        else:
            self.markov_head = None
            logger.info(
                "[treedspark-dflare] markov_rank<=0; skipping Markov head "
                "construction (draft will run without Markov bias)."
            )

        # ---- TreeFlash hidden-states correction (optional) ----
        self.enable_hidden_correction = bool(
            getattr(config, "enable_hidden_correction", False)
        )
        if self.enable_hidden_correction:
            intermediate = getattr(config, "hidden_correction_intermediate_size", None)
            if intermediate is None:
                # Training-side default when the field is None: match hidden_size.
                intermediate = int(config.hidden_size)
            else:
                intermediate = int(intermediate)
            self.hidden_correction = HiddenStatesCorrection(
                hidden_size=int(config.hidden_size),
                embed_size=int(config.hidden_size),
                intermediate_size=intermediate,
                rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                prefix=maybe_prefix(prefix, "hidden_correction"),
            )
        else:
            self.hidden_correction = None


class Qwen3TreeDSparkDFlareForCausalLM(Qwen3DSparkForCausalLM):
    """Top-level TreeFlash + DSpark + DFlare draft model.

    Structure vs. its bases:
      * inner model is ``Qwen3TreeDSparkDFlareModel`` (DFlare backbone +
        Markov head + optional HiddenStatesCorrection);
      * ``compute_draft_logits`` is unchanged (LM head over ``hidden``); the
        hidden correction is applied by the speculator per Markov step, right
        before this call, via ``apply_hidden_correction``.
      * ``load_weights`` extends both parents to route the DFlare
        target-K/V weights, the ``layer_fusion_weights``, the Markov head,
        and the new ``hidden_correction.*`` weights into the right params.
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
        self.model = Qwen3TreeDSparkDFlareModel(
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

    # ------------------------------------------------------------------
    # combine_hidden_states override
    # ------------------------------------------------------------------
    # DFlare replaces DFlash's ``fc`` (T-layer collapse) with per-layer fusion
    # that runs inside ``precompute_and_store_context_kv``. The inner
    # ``Qwen3TreeDSparkDFlareModel`` (via ``DFlareQwen3Model``) deletes
    # ``self.model.fc`` in ``__init__``, so we MUST NOT fall back to
    # ``DFlashQwen3ForCausalLM.combine_hidden_states`` (which calls
    # ``self.model.fc(...)`` and raises AttributeError). Mirror
    # ``DFlareQwen3ForCausalLM.combine_hidden_states``: a no-op that hands
    # the raw ``[N, T*D]`` concatenation to the speculator.
    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    # ------------------------------------------------------------------
    # Hidden correction API used by DSparkSpeculator._sample_sequential
    # ------------------------------------------------------------------
    def has_hidden_correction(self) -> bool:
        return getattr(self.model, "hidden_correction", None) is not None

    def apply_hidden_correction(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the TreeFlash hidden-states correction.

        Args:
            hidden_states: ``[N, hidden_size]`` output hidden of the drafter
                for the positions about to be sampled.
            prev_token_ids: ``[N]`` LongTensor of the previously sampled
                (teacher-forced at step 0) token ids for each of the N rows.

        Returns:
            Corrected hidden state, same shape as ``hidden_states``. If
            hidden correction is disabled, returns ``hidden_states`` unchanged.
        """
        hc = getattr(self.model, "hidden_correction", None)
        if hc is None:
            return hidden_states
        # Reuse the same embedding table the backbone uses (which is shared
        # with the target model when the draft ships no embed_tokens).
        prev_embeds = self.model.embed_input_ids(prev_token_ids)
        return hc(hidden_states, prev_embeds)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Merge the DFlare and DSpark loader rules, plus new remaps for the
        TreeFlash hidden_correction module.

        Training-side checkpoint keys we care about (see the checkpoint
        listing at iter_0018433_hf):

          embed_tokens.weight
          norm.weight
              -> model.norm.weight (via ``final_norm`` alias)
          hidden_norm.weight
              -> model.hidden_norm.weight
          layer_fusion_weights
              -> model.layer_fusion_weights
          layers.<i>.self_attn.{q,k,v}_proj.weight
              -> qkv_proj (stacked)
          layers.<i>.self_attn.{k,v}_proj_target.weight
              -> DFlare per-layer target proj (loaded directly)
          layers.<i>.self_attn.{q,k}_norm.weight
              -> unchanged
          layers.<i>.self_attn.o_proj.weight
              -> unchanged
          layers.<i>.mlp.{gate,up}_proj.weight
              -> gate_up_proj (stacked)
          layers.<i>.mlp.down_proj.weight
              -> unchanged
          layers.<i>.{input,post_attention}_layernorm.weight
              -> unchanged
          markov_head.markov_w{1,2}.weight
              -> model.markov_head.markov_w{1,2}.weight
          hidden_correction.hidden_norm.weight
              -> model.hidden_correction.hidden_norm.weight
          hidden_correction.embed_norm.weight
              -> model.hidden_correction.embed_norm.weight
          hidden_correction.gate_proj.weight
              -> model.hidden_correction.gate_up_proj (shard 0)
          hidden_correction.up_proj.weight
              -> model.hidden_correction.gate_up_proj (shard 1)
          hidden_correction.down_proj.weight
              -> model.hidden_correction.down_proj.weight
        """
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )

        # ---- First pass: normalise names & split into two buckets ----
        #
        # bucket A: goes through the DFlare-aware stacked-params loop
        # (handles qkv_proj / gate_up_proj / target K,V bypass).
        # bucket B: goes through AutoWeightsLoader (embed / lm_head /
        # markov / hidden_correction non-gate weights).
        #
        # We only pre-remap:
        #   context_norm    -> hidden_norm   (older DFlare training exports)
        #   final_norm      -> norm          (drafter output norm)
        #   fc.*            -> dropped       (DFlare replaces with per-layer fusion)
        #   context_proj.*  -> dropped       (DFlare replaces with per-layer fusion)
        #   confidence_head -> dropped       (not wired into inference)
        #   mask_embedding  -> dropped       (checkpoint has no such tensor)
        #   d2t             -> draft_id_to_target_id
        #   t2d             -> dropped

        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        includes_lm_head = False
        has_markov_head = getattr(self.model, "markov_head", None) is not None

        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "confidence_head" in name:
                # Not consumed by inference; drop it.
                continue
            if "context_proj" in name:
                continue
            if name.startswith("fc.") or ".fc." in name:
                # DFlash's collapse layer; unused by DFlare.
                continue
            if not has_markov_head and ("markov_head" in name or "markov_w" in name):
                # Markov head disabled at model-build time (markov_rank<=0);
                # drop any Markov weights that happen to live in the checkpoint.
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name

            # Aliasing rules for training-side names that differ from vLLM.
            if name == "model.context_norm.weight":
                name = "model.hidden_norm.weight"
            if name == "model.final_norm.weight":
                name = "model.norm.weight"

            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name and "markov" not in name:
                includes_lm_head = True

            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # ---- Second pass: DFlare-aware per-parameter loader for the
        #                    stacked qkv / gate_up projections.
        #
        # This mirrors DFlareQwen3Model.load_weights so the fused
        # ``qkv_proj`` and ``gate_up_proj`` get their shards correctly,
        # while ``k_proj_target`` / ``v_proj_target`` (which live at the
        # layer level) and the new hidden_correction gate_up_proj bypass
        # the substring-based stacked mapping. Remaining tensors (norms,
        # markov, hidden_correction.{hidden,embed}_norm, down_proj) fall
        # through to AutoWeightsLoader below.
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_param_names: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        remaining_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in model_weights.items():
            if "scale" in name:
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped

            if "attention_sink_bias" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                heads_per_rank = loaded_weight.shape[0] // tp_size
                head_start = tp_rank * heads_per_rank
                narrow = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow)
                loaded_param_names.add(name)
                continue

            # DFlare per-layer target K/V: match ``.k_proj_target`` /
            # ``.v_proj_target`` BEFORE the ``.k_proj`` / ``.v_proj``
            # substring rule would false-match them.
            if ".k_proj_target" in name or ".v_proj_target" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_param_names.add(name)
                continue

            # Route hidden_correction's ``gate_proj`` / ``up_proj`` into the
            # fused ``gate_up_proj`` via the standard shard mapping. The
            # training-side keys are
            # ``model.hidden_correction.gate_proj.weight`` etc.; the target
            # vLLM param is ``model.hidden_correction.gate_up_proj.weight``.
            #
            # This is the SAME rule that already handles the per-layer MLP
            # gate_up fusion; we just fall through to the loop below.
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_param_names.add(mapped)
                matched = True
                break
            if matched:
                continue

            # Everything else goes through the generic per-parameter loader.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_param_names.add(name)
            else:
                # Unknown / unused checkpoint tensor; keep it around only to
                # let AutoWeightsLoader raise if it's expected somewhere.
                remaining_weights.append((name, loaded_weight))

        # ---- Third pass (fallback): run AutoWeightsLoader on anything we
        #      didn't match by name above. This is a safety net; in practice
        #      the loop above already handles every checkpoint tensor we
        #      have. AutoWeightsLoader will complain loudly if a *required*
        #      param is still missing, which is the diagnostic we want.
        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        skip_substrs.append("fc.")
        if not has_markov_head:
            skip_substrs.append("markov")

        if remaining_weights:
            loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
            loader.load_weights(iter(remaining_weights))

        # Build the fused KV buffers used by precompute_and_store_context_kv.
        self.model._build_fused_kv_buffers()


__all__ = [
    "HiddenStatesCorrection",
    "Qwen3TreeDSparkDFlareModel",
    "Qwen3TreeDSparkDFlareForCausalLM",
]
