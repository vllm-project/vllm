# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The parallel backbone is a standard Qwen3 decoder stack reused from the
DFlash Qwen3 draft (see qwen3_dflash.py). DSpark adds:
  * ``markov_head``: low-rank V x r / r x V transition bias added to the base
    logits, sampled left-to-right by the speculator (the sequential stage).

DSparkMarkovHead is shared with the DSV4-style DSpark model.
"""

import functools
import os
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


@functools.cache
def _get_hpc_module():
    """Import hpc-ops lazily so model discovery does not initialize CUDA."""
    try:
        import hpc
    except (ImportError, OSError) as error:
        logger.warning_once("[dspark] hpc module is unavailable: %s", error)
        return None
    required_ops = ("fused_rmsnorm_blockwise_quant", "add")
    if not all(hasattr(hpc, name) for name in required_ops):
        logger.warning_once(
            "[dspark] hpc module does not provide all required correction ops."
        )
        return None
    return hpc


class HpcHiddenCorrectionMixin:
    """HPC RMSNorm/add helpers shared by hidden-correction variants."""

    def _init_hpc_correction(self, *, default_enabled: bool = True) -> None:
        self._use_hpc_correction = (
            os.getenv("VLLM_DSPARK_HPC_CORRECTION", "1" if default_enabled else "0")
            != "0"
        )
        self._hpc_norm_disabled = False
        self._hpc_add_disabled = False

    def _norm_with_hpc(self, norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        if not self._use_hpc_correction or self._hpc_norm_disabled:
            return norm(x)
        if not x.is_cuda or x.dtype != torch.bfloat16 or norm.weight.dtype != x.dtype:
            return norm(x)

        hpc = _get_hpc_module()
        if hpc is None:
            self._hpc_norm_disabled = True
            return norm(x)
        try:
            # hpc-ops 5.2.0 assumes dense row-major input but does not reject
            # strided views such as hidden_per_step[:, i]. Materialize those
            # views to avoid silently normalizing values from adjacent steps.
            hpc_input = x if x.is_contiguous() else x.contiguous()
            return hpc.fused_rmsnorm_blockwise_quant(
                hpc_input,
                norm.weight.data,
                norm.variance_epsilon,
                False,
            )
        except (RuntimeError, TypeError) as error:
            self._hpc_norm_disabled = True
            logger.warning_once(
                "[dspark] hpc RMSNorm failed; falling back to vLLM RMSNorm: %s",
                error,
            )
            return norm(x)

    def _residual_add_with_hpc(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta.to(hidden_states.dtype)
        if (
            not self._use_hpc_correction
            or self._hpc_add_disabled
            or not hidden_states.is_cuda
        ):
            return hidden_states + delta

        # hpc-ops 5.2.0 exposes add for FP32 only. Hidden correction normally
        # runs in BF16, so avoid a known failing dispatch and retain torch.add.
        if hidden_states.dtype != torch.float32:
            return hidden_states + delta

        hpc = _get_hpc_module()
        if hpc is None:
            self._hpc_add_disabled = True
            return hidden_states + delta
        try:
            return hpc.add(hidden_states, delta)
        except (RuntimeError, TypeError) as error:
            self._hpc_add_disabled = True
            logger.warning_once(
                "[dspark] hpc add failed; falling back to torch add: %s", error
            )
            return hidden_states + delta


class PositionAdaptiveAlpha(nn.Module):
    """Per-in-block-position Markov / correction strength ``alpha_i``.

    Implements ``alpha_i = alpha_max * sigmoid(w_i)`` with a learnable logit
    vector of length ``block_size``. Used at inference as
    ``logits_i += alpha_i * markov_bias_i``.
    """

    def __init__(
        self,
        *,
        block_size: int,
        alpha_max: float = 1.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError(
                f"PositionAdaptiveAlpha requires block_size > 0, got {block_size}."
            )
        self.block_size = int(block_size)
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(
            torch.empty(self.block_size), requires_grad=False
        )

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def alpha_at(self, step: int) -> torch.Tensor:
        if step < 0 or step >= self.block_size:
            raise IndexError(
                f"Markov pos-alpha step {step} out of range [0, {self.block_size})."
            )
        return self.alpha()[step]


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` embeds the previously sampled token (target vocab,
    ``vocab_size``); ``markov_w2`` projects it to a draft-vocab bias
    (``draft_vocab_size``) added to the base draft logits. The two sizes
    coincide for full-vocab drafts.

    Optional ``pos_alpha`` scales the bias per in-block draft position when
    the checkpoint was trained with ``markov_pos_adaptive=True``.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        *,
        pos_adaptive: bool = False,
        block_size: int | None = None,
        alpha_max: float = 1.0,
        replicated: bool = False,
    ) -> None:
        super().__init__()
        self.replicated = replicated
        if replicated:
            # The low-rank Markov head is small enough to keep on every TP
            # rank. This removes one embedding all-reduce and one full-vocab
            # logits all-gather from every sequential Markov step.
            self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
            self.markov_w2 = ReplicatedLinear(
                markov_rank,
                draft_vocab_size,
                bias=False,
                prefix=maybe_prefix(prefix, "markov_w2"),
                return_bias=False,
            )
        else:
            self.markov_w1 = VocabParallelEmbedding(
                vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
            )
            self.markov_w2 = ParallelLMHead(
                draft_vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_w2"),
            )
        self.pos_alpha: PositionAdaptiveAlpha | None = None
        if pos_adaptive:
            if block_size is None or int(block_size) <= 0:
                raise ValueError(
                    "DSparkMarkovHead pos_adaptive requires a positive block_size."
                )
            self.pos_alpha = PositionAdaptiveAlpha(
                block_size=int(block_size),
                alpha_max=float(alpha_max),
                prefix=maybe_prefix(prefix, "pos_alpha"),
            )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor,
        step: int | None = None,
    ) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        if self.replicated:
            bias = self.markov_w2(markov_embed)
        else:
            bias = logits_processor(self.markov_w2, markov_embed)
        if self.pos_alpha is not None and step is not None:
            bias = bias * self.pos_alpha.alpha_at(step).to(dtype=bias.dtype)
        return bias


class HiddenStatesCorrection(HpcHiddenCorrectionMixin, nn.Module):
    """TreeFlash hidden-states correction (formula (1)).

    Applies a residual SwiGLU to the drafter's output hidden state,
    conditioned on the previous token's embedding::

        h' = h + down_proj( SiLU(gate_proj(cat[norm_h(h), norm_e(e_prev)]))
                          * up_proj(cat[norm_h(h), norm_e(e_prev)]) )

    The training-side module (see ``torchspec/models/draft/dspark.py``) uses
    two independent ``nn.Linear`` layers ``gate_proj`` and ``up_proj``. We use
    a fused ``MergedColumnParallelLinear`` here for TP efficiency; the load
    path remaps the per-head training checkpoint entries onto its shard IDs.

    The down-projection is expected to be zero-initialized in the checkpoint
    (residual zero-init), so before training kicks in the correction is a
    no-op and the model reduces exactly to the base draft + Markov head.
    """

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.intermediate_size = intermediate_size

        # Independent RMSNorm on each stream so h and e enter the SwiGLU on
        # comparable scales (they live in different embedding spaces even
        # though for the tied-embedding drafter their dims match).
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = RMSNorm(embed_size, eps=rms_norm_eps)

        # Fused gate+up projection (matches Qwen2/3 MLP layout in vLLM so the
        # existing MergedColumnParallelLinear shard logic can be reused).
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size + embed_size,
            [intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        self._init_hpc_correction()

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the residual correction.

        Args:
            hidden_states: ``[N, hidden_size]`` drafter output hidden state.
            prev_token_embeds: ``[N, embed_size]`` previous-token embedding
                (same leading shape as ``hidden_states``).

        Returns:
            Corrected hidden state, ``[N, hidden_size]``.
        """
        h = self._norm_with_hpc(self.hidden_norm, hidden_states)
        e = self._norm_with_hpc(
            self.embed_norm, prev_token_embeds.to(hidden_states.dtype)
        )
        cat = torch.cat([h, e], dim=-1)
        gate_up, _ = self.gate_up_proj(cat)
        x = self.act_fn(gate_up)
        delta, _ = self.down_proj(x)
        return self._residual_add_with_hpc(hidden_states, delta)


class HiddenStatesCorrectionLowRank(nn.Module):
    """Low-rank bottleneck variant of :class:`HiddenStatesCorrection`.

    Mirrors the training-side ``HiddenStatesCorrectionLowRank`` (see
    ``torchspec/models/draft/dspark.py``): a lightweight adapter that
    down-projects the concatenated ``[norm_h(h) :: norm_e(e_prev)]`` (dim
    ``2 * d_model``) into a low-rank width ``r`` (``intermediate_size``),
    applies a plain ``SiLU`` (no gating), then up-projects back to
    ``hidden_size``::

        x = cat[norm_h(h), norm_e(e_prev)]
        z = SiLU(down_proj(x))           # A: 2d -> r
        delta = up_proj(z)               # B: r -> d
        h' = h + delta

    Unlike :class:`HiddenStatesCorrection`, there is NO fused ``gate_up_proj``:
    the training checkpoint stores two independent matrices ``down_proj``
    (r x 2d) and ``up_proj`` (d x r), and we keep the same layout here.

    The up-projection is expected to be zero-initialized in the checkpoint
    (residual zero-init), so before training kicks in the correction is a
    no-op and the model reduces exactly to the base draft + Markov head.
    """

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.intermediate_size = intermediate_size

        # Independent RMSNorm on each stream so h and e enter the adapter on
        # comparable scales, matching the SwiGLU variant.
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = RMSNorm(embed_size, eps=rms_norm_eps)

        # A: 2d -> r (column-parallel: shard the low-rank width across TP).
        self.down_proj = ColumnParallelLinear(
            hidden_size + embed_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.down_proj",
        )
        # B: r -> d (row-parallel: input is the sharded low-rank code, output
        # is all-reduced back to full hidden_size).
        self.up_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.up_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the residual low-rank correction.

        Args:
            hidden_states: ``[N, hidden_size]`` drafter output hidden state.
            prev_token_embeds: ``[N, embed_size]`` previous-token embedding
                (same leading shape as ``hidden_states``).

        Returns:
            Corrected hidden state, ``[N, hidden_size]``.
        """
        h = self.hidden_norm(hidden_states)
        e = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        x = torch.cat([h, e], dim=-1)
        z, _ = self.down_proj(x)
        z = torch.nn.functional.silu(z)
        delta, _ = self.up_proj(z)
        return hidden_states + delta.to(hidden_states.dtype)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov head."""

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
        # it to <= 0), we skip constructing the Markov head entirely and the
        # sampler falls back to plain draft-logits (no Markov bias).
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_rank = int(getattr(config, "markov_rank", 0) or 0)
        if markov_rank > 0:
            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                draft_vocab_size,
                config.markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
                pos_adaptive=bool(getattr(config, "markov_pos_adaptive", False)),
                block_size=getattr(config, "block_size", None),
                alpha_max=float(getattr(config, "markov_alpha_max", 1.0)),
            )
        else:
            self.markov_head = None
            logger.info(
                "[dspark] markov_rank<=0; skipping Markov head construction "
                "(draft will run without Markov bias)."
            )

        # ---- TreeFlash hidden-states correction (optional) ----
        # Enabled via ``enable_hidden_correction`` in the draft config; when
        # off, the module is not constructed and ``has_hidden_correction``
        # returns False so DSparkSpeculator skips the correction path.
        # Down-projection is expected to be zero-initialized in the checkpoint,
        # so the correction is a no-op at init and the model reduces exactly
        # to DFlash + Markov.
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

            hc_type = str(getattr(config, "hidden_correction_type", "swiglu")).lower()
            if hc_type == "low_rank":
                self.hidden_correction = HiddenStatesCorrectionLowRank(
                    hidden_size=int(config.hidden_size),
                    embed_size=int(config.hidden_size),
                    intermediate_size=intermediate,
                    rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    prefix=maybe_prefix(prefix, "hidden_correction"),
                )
            else:
                if hc_type != "swiglu":
                    logger.warning_once(
                        "Unsupported hidden_correction_type=%r; falling back "
                        "to SwiGLU.",
                        hc_type,
                    )
                self.hidden_correction = HiddenStatesCorrection(
                    hidden_size=int(config.hidden_size),
                    embed_size=int(config.hidden_size),
                    intermediate_size=intermediate,
                    rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    prefix=maybe_prefix(prefix, "hidden_correction"),
                )
        else:
            self.hidden_correction = None


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DSparkModel(
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

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Draft-vocab logits without the d2t scatter: the speculator adds the
        # Markov bias in draft space, then remaps via map_draft_to_target.
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # Map draft-vocab ids to target ids (identity for full-vocab drafts).
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def has_markov(self) -> bool:
        return getattr(self.model, "markov_head", None) is not None

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(
        self,
        markov_embed: torch.Tensor,
        step: int | None = None,
    ) -> torch.Tensor:
        return self.model.markov_head.bias(
            markov_embed, self.logits_processor, step=step
        )

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        has_markov_head = getattr(self.model, "markov_head", None) is not None

        # Detect low-rank hidden_correction: it has ``down_proj`` (2d->r) and
        # ``up_proj`` (r->d) as two independent matrices, and (critically) NO
        # fused ``gate_up_proj``. The DFlashQwen3Model.load_weights stacked
        # mapping would otherwise rewrite ``.up_proj -> .gate_up_proj`` and
        # crash on ``KeyError`` because the target param does not exist.
        hc = getattr(self.model, "hidden_correction", None)
        hc_is_low_rank = hc is not None and not hasattr(hc, "gate_up_proj")

        for name, loaded_weight in weights:
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if not has_markov_head and ("markov_head" in name or "markov_w" in name):
                # Markov head disabled at model-build time (markov_rank<=0);
                # drop any Markov weights the checkpoint might still ship.
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens / has_own_lm_head so load_dspark_model
            # knows whether to keep these or alias the target's.
            process_eagle_weight(self, name)

        # ---- Pre-load bypass: low-rank hidden_correction ----
        # Manually load ``model.hidden_correction.{down,up}_proj.weight`` here,
        # then remove them from ``model_weights`` so AutoWeightsLoader (which
        # dispatches to DFlashQwen3Model.load_weights via recursion) does not
        # see them and does not run its stacked ``.up_proj -> .gate_up_proj``
        # rewrite that assumes a fused gate_up_proj target.
        if hc_is_low_rank:
            params_dict = dict(self.named_parameters())
            for lr_name in (
                "model.hidden_correction.down_proj.weight",
                "model.hidden_correction.up_proj.weight",
            ):
                if lr_name not in model_weights:
                    continue
                if lr_name not in params_dict:
                    # Should not happen given hc_is_low_rank, but stay safe.
                    continue
                w = model_weights.pop(lr_name)
                param = params_dict[lr_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, w)
            logger.info(
                "[dspark-load] low-rank hidden_correction detected; "
                "pre-loaded down_proj/up_proj to bypass stacked mapping."
            )

        # mask_embedding is an unused placeholder param; DSpark masks via the vocab row.
        # confidence_head is not wired into inference yet; skip its weights.
        # embed_tokens / lm_head are optional; when omitted they are shared from
        # the target by load_dspark_model, so skip the unloaded params here.
        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not has_markov_head:
            skip_substrs.append("markov")
        if hc_is_low_rank:
            # Also tell AutoWeightsLoader to skip these params so it does not
            # complain about "unexpected weights" / missing (they've already
            # been loaded above via the direct pre-load bypass).
            skip_substrs.append("hidden_correction.down_proj")
            skip_substrs.append("hidden_correction.up_proj")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
