# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# LiLiCorr head adapted from sgl-project/sglang PR #37462 (Apache-2.0).
"""DFlash backbone with the LiLiCorr candidate-lattice correlator."""

from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.linear import (
    LinearBase,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .qwen3_dflash2 import DFlash2Qwen3DecoderLayer
from .utils import maybe_prefix


class LiLiCorrLatticeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible "
                f"by num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        # Exported lattice QKV parameters stay in the model's floating dtype.
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size), requires_grad=False
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty(3 * hidden_size), requires_grad=False
        )
        self.out_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "out_proj"),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # Attention and MMEncoderAttention cannot accept this learned per-head
        # additive bias. The full candidate lattice has no KV-cache state.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
        )
        return self.out_proj(
            out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        )


class LiLiCorrMLP(nn.Sequential):
    """Biased SiLU MLP retaining the checkpoint's numeric module names."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        normalize_input: bool = False,
    ) -> None:
        modules: list[nn.Module] = [nn.LayerNorm(input_size)] if normalize_input else []
        input_index = len(modules)
        modules.extend(
            [
                ReplicatedLinear(
                    input_size,
                    intermediate_size,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, str(input_index)),
                ),
                get_act_fn("silu"),
                ReplicatedLinear(
                    intermediate_size,
                    output_size,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, str(input_index + 2)),
                ),
            ]
        )
        super().__init__(*modules)
        self._input_index = input_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._input_index:
            x = self[0](x)
        x = self[self._input_index](x)
        down_proj = cast(ReplicatedLinear, self[-1])
        x = maybe_fused_act_quant(self[self._input_index + 1], x, down_proj)
        return down_proj(x)


class LiLiCorrLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = LiLiCorrLatticeAttention(
            hidden_size, num_heads, quant_config, maybe_prefix(prefix, "attn")
        )
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # The checkpoint uses a biased, nongated MLP, unlike Qwen's SwiGLU.
        self.mlp = LiLiCorrMLP(
            hidden_size,
            mlp_hidden_size,
            hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mlp"),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), attention_bias
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


@support_torch_compile
class LiLiCorrHead(nn.Module):
    num_candidate_features = 5

    def __init__(
        self,
        *,
        model_hidden_size: int,
        block_size: int,
        rms_norm_eps: float,
        config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config["lilicorr_hidden_size"] or model_hidden_size
        self.block_size = block_size
        self.num_candidate_slots = block_size - 1
        self.candidate_topk = config["lilicorr_candidate_topk"]
        if (
            self.candidate_topk <= 0
            or self.candidate_topk > 16
            or self.candidate_topk & (self.candidate_topk - 1)
        ):
            raise ValueError("LiLiCorr candidate_topk must be a power of two <= 16.")
        self.hidden_size = hidden_size
        self.num_heads = config["lilicorr_num_heads"]
        self.mlp_ratio = config["lilicorr_mlp_ratio"]
        self.factor_dim = config["lilicorr_factor_dim"]
        self.vector_eps = config["lilicorr_vector_eps"]
        self.logit_scale = config["lilicorr_logit_scale"]
        # Candidate IDs and embeddings are replicated across TP ranks, so each
        # rank scores the same lattice without further head collectives.
        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else ReplicatedLinear(
                model_hidden_size,
                hidden_size,
                return_bias=False,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "token_proj"),
            )
        )
        self.pass_hidden_proj = ReplicatedLinear(
            model_hidden_size,
            hidden_size,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "pass_hidden_proj"),
        )
        self.feature_mlp = LiLiCorrMLP(
            self.num_candidate_features,
            hidden_size,
            hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "feature_mlp"),
            normalize_input=True,
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_candidate_slots, 1, hidden_size)
        )
        self.rank_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, self.candidate_topk, hidden_size)
        )
        self.relative_slot_bias = nn.Parameter(
            torch.zeros(self.num_heads, 2 * self.block_size - 1)
        )
        self.same_slot_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.context_proj = ReplicatedLinear(
            model_hidden_size,
            hidden_size,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    rms_norm_eps=rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i}"),
                )
                for i in range(config["lilicorr_num_layers"])
            ]
        )
        self.output_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.factor_input_proj = ReplicatedLinear(
            hidden_size * 3,
            hidden_size,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "factor_input_proj"),
        )
        # These are latent edge factors, not vocabulary projections. A candidate's
        # outgoing vector scores its compatibility with each next-slot candidate's
        # incoming vector; the anchor supplies the outgoing vector for slot zero.
        self.out_head = ReplicatedLinear(
            hidden_size,
            self.factor_dim,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "out_head"),
        )
        self.in_head = ReplicatedLinear(
            hidden_size,
            self.factor_dim,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "in_head"),
        )
        self.anchor_out_head = ReplicatedLinear(
            hidden_size,
            self.factor_dim,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "anchor_out_head"),
        )
        self._fused_edge_weight: torch.Tensor | None = None
        self._fused_edge_bias: torch.Tensor | None = None
        self._factor_input_splits: tuple[torch.Tensor, ...]
        self._attn_bias: torch.Tensor | None = None
        self._rank_frac_col: torch.Tensor
        self._is_top1_col: torch.Tensor

    @torch.no_grad()
    def materialize_inference_buffers(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Build inference-only views/copies after checkpoint weights are loaded."""
        topk = self.candidate_topk
        self._attn_bias = self._build_attention_bias(device=device, dtype=dtype)
        # Both edge heads consume the same factor_hidden. Stacking their output
        # rows computes both projections in one linear call; score() splits and
        # normalizes each vector separately. Do this once after loading, keeping
        # the original parameter names for checkpoint compatibility.
        self._fused_edge_weight = (
            torch.cat([self.out_head.weight, self.in_head.weight], dim=0)
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        self._fused_edge_bias = (
            torch.cat([self.out_head.bias, self.in_head.bias], dim=0)
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        # The trained factor input is [x, anchor, x*anchor] for every candidate x.
        # Splitting W along its input dimension lets score() project the anchor
        # once per request and broadcast it, without materializing the concatenated
        # input. The three contributions are summed with the original bias once.
        weight = self.factor_input_proj.weight
        hdim = self.hidden_size
        self._factor_input_splits = (
            weight[:, :hdim].contiguous(),
            weight[:, hdim : 2 * hdim].contiguous(),
            weight[:, 2 * hdim :].contiguous(),
        )
        if topk > 1:
            rank_frac = torch.arange(topk, device=device, dtype=torch.float32).view(
                1, 1, topk
            ) / float(topk - 1)
        else:
            rank_frac = torch.zeros(1, 1, topk, device=device, dtype=torch.float32)
        is_top1 = torch.zeros(1, 1, topk, device=device, dtype=torch.float32)
        is_top1[..., 0] = 1.0
        self._rank_frac_col = rank_frac.contiguous()
        self._is_top1_col = is_top1.contiguous()

    def _build_attention_bias(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        topk = self.candidate_topk
        slot_ids = torch.arange(
            self.num_candidate_slots, device=device, dtype=torch.long
        ).repeat_interleave(topk)
        # Flattened lattice index = slot * topk + candidate rank. All candidates
        # in a slot therefore share a position, including distinct candidate pairs
        # within that slot. Each head learns a relative-slot bias plus an extra
        # same-slot term; this is an additive score bias, not a causal mask.
        # Center the lookup at the trained block_size-1 even for shorter drafts;
        # score() selects the matching prefix of the resulting matrix.
        rel = slot_ids.view(-1, 1) - slot_ids.view(1, -1)
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(
            dtype=bias.dtype
        ) * self.same_slot_bias.view(-1, 1, 1)
        return bias.to(device=device, dtype=dtype).contiguous()

    def score(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._attn_bias is None:
            raise RuntimeError(
                "Call materialize_inference_buffers() after loading the LiLiCorr head."
            )
        bsz, n_slots, topk = candidate_log_probs.shape
        if not 1 <= n_slots <= self.num_candidate_slots:
            raise ValueError(
                f"LiLiCorr supports 1..{self.num_candidate_slots} candidate slots, "
                f"got {n_slots}."
            )
        if topk != self.candidate_topk:
            raise ValueError(
                f"LiLiCorr expects candidate_topk={self.candidate_topk}, got {topk}."
            )
        proj_dtype = self.slot_embedding.dtype
        if token_embeddings.dtype != proj_dtype:
            token_embeddings = token_embeddings.to(proj_dtype)
        if pass_hidden.dtype != proj_dtype:
            pass_hidden = pass_hidden.to(proj_dtype)
        token_states = self.token_proj(token_embeddings)
        pass_states = self.pass_hidden_proj(pass_hidden).unsqueeze(-2)
        log_probs = candidate_log_probs.float()
        features = torch.stack(
            [
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.max(dim=-1, keepdim=True).values,
                self._rank_frac_col.expand_as(log_probs),
                self._is_top1_col.expand_as(log_probs),
            ],
            dim=-1,
        )
        hidden_states = token_states + pass_states
        hidden_states = hidden_states + self.feature_mlp(
            features.to(dtype=token_states.dtype)
        )
        hidden_states = hidden_states + self.slot_embedding[:, 0, :n_slots]
        hidden_states = hidden_states + self.rank_embedding[:, 0]
        hidden_states = hidden_states.reshape(bsz, n_slots * topk, self.hidden_size)
        anchor_state = self.context_proj(anchor_hidden)
        anchor_state = anchor_state * anchor_valid[:, None].to(anchor_state.dtype)
        # Shorter drafts use the learned prefix; keep checkpoint parameter shapes.
        lattice = n_slots * topk
        attention_bias = self._attn_bias[None, :, :lattice, :lattice]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        hidden_states = self.output_norm(hidden_states).reshape(
            bsz, n_slots, topk, self.hidden_size
        )
        anchor_state = self.anchor_norm(anchor_state)
        w_self, w_anchor, w_cross = self._factor_input_splits
        anchor_row = anchor_state[:, None, None, :]
        pre = F.linear(hidden_states, w_self, self.factor_input_proj.bias)
        pre = pre + F.linear(anchor_row, w_anchor)
        pre = pre + F.linear(hidden_states * anchor_row, w_cross)
        factor_hidden = F.silu(pre)
        edges = F.linear(factor_hidden, self._fused_edge_weight, self._fused_edge_bias)
        out_vec, in_vec = F.normalize(
            edges.unflatten(-1, (2, self.factor_dim)), dim=-1, eps=self.vector_eps
        ).unbind(-2)
        anchor_out = F.normalize(
            self.anchor_out_head(anchor_state), dim=-1, eps=self.vector_eps
        )
        # start_scores[b, j] scores anchor -> candidate j in slot zero.
        # pair_scores[b, s, i, j] scores candidate i in slot s -> candidate j
        # in slot s+1. Normalized factor dot products encode learned compatibility;
        # forward() scales them into logits for the candidate-path selector.
        start_scores = (anchor_out[:, None, :] * in_vec[:, 0, :, :]).sum(dim=-1)
        pair_scores = torch.matmul(out_vec[:, :-1], in_vec[:, 1:].transpose(-1, -2))
        return (start_scores, pair_scores)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> torch.Tensor:
        start, pairs = self.score(
            token_embeddings=token_embeddings,
            candidate_log_probs=candidate_log_probs,
            pass_hidden=pass_hidden,
            anchor_hidden=anchor_hidden,
            anchor_valid=anchor_valid,
        )
        start = self.logit_scale * start.float()
        pairs = self.logit_scale * pairs.float()
        # The shared walk starts at predecessor index zero; all first rows agree.
        first = start[:, None, None, :].expand(-1, 1, self.candidate_topk, -1)
        return torch.cat((first, pairs), dim=1)


class LiLiCorr(DFlashQwen3Model):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        draft_config = config.dflash_config
        if not draft_config.get("lilicorr_enabled", True):
            raise ValueError("LiLiCorr requires lilicorr_enabled=True.")
        block_size = draft_config.get("block_size", getattr(config, "block_size", 0))
        if not 1 <= spec.num_speculative_tokens < block_size:
            raise ValueError(
                "LiLiCorr requires 1 <= num_speculative_tokens < trained block_size."
            )
        taps = draft_config.get("conv_kernel_size", 0)
        groups = draft_config.get("conv_group_size", 0)
        if taps < 0 or groups < 0 or bool(taps) != bool(groups):
            raise ValueError(
                "LiLiCorr convolution geometry must enable both taps and group size."
            )
        if taps:
            self.decoder_layer_cls = DFlash2Qwen3DecoderLayer
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        with set_model_tag("lilicorr_head"):
            self.lilicorr = LiLiCorrHead(
                model_hidden_size=config.hidden_size,
                block_size=block_size,
                rms_norm_eps=config.rms_norm_eps,
                config=draft_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lilicorr"),
            )


class LiLiCorrForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = LiLiCorr

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        spec = vllm_config.speculative_config
        assert spec is not None
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.draft_id_to_target_id is not None:
            raise ValueError("LiLiCorr candidates require the full target vocabulary.")
        self.has_own_lm_head = bool(getattr(self.config, "has_own_lm_head", False))
        # The correlator consumes raw candidate-head log-softmax features.
        self.candidate_logits_processor = LogitsProcessor(self.config.vocab_size)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head,
            hidden_states,
            self.model.lilicorr.candidate_topk,
            return_log_probs=True,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        quantized_metadata: set[str] = set()
        for name, module in self.named_modules():
            name = name.removeprefix("model.")
            if not isinstance(module, (LinearBase, ParallelLMHead)) or isinstance(
                module.quant_method, UnquantizedLinearMethod
            ):
                continue
            required = {"weight", "bias"}
            method = module.quant_method
            if (
                isinstance(method, ModelOptLinearMethod)
                and method.spec.weight == kNvfp4Static
            ):
                # NVFP4 scales are trained checkpoint data. Only W4A16's
                # deprecated, unused input_scale may be absent.
                required.update(("weight_scale", "weight_scale_2"))
                if method.spec.activation is not None:
                    required.add("input_scale")
            quantized_metadata.update(
                f"{name}.{parameter}"
                for parameter, _ in module.named_parameters(recurse=False)
                if parameter not in required
            )
        expected = {
            name.removeprefix("model.")
            for name, _ in self.named_parameters()
            if (
                name.startswith("model.lilicorr.")
                or ".attention_conv." in name
                or ".mlp_conv." in name
                or (self.has_own_lm_head and name.startswith("lm_head."))
            )
            and name.removeprefix("model.") not in quantized_metadata
        }
        seen: set[str] = set()

        def normalized_weights():
            for name, value in weights:
                name = name.removeprefix("model.")
                seen.add(name)
                yield name, value

        super().load_weights(normalized_weights())
        missing = expected - seen
        if missing:
            raise ValueError(
                f"LiLiCorr checkpoint coverage mismatch: missing={sorted(missing)}"
            )
        head = self.model.lilicorr
        parameter = head.slot_embedding
        head.materialize_inference_buffers(parameter.device, parameter.dtype)
