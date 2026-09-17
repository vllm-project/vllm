# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# LiLiCorr head adapted from sgl-project/sglang PR #37462 (Apache-2.0).
"""DFlash backbone with the LiLiCorr candidate-lattice correlator."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .qwen3_dflash2 import DFlash2Qwen3DecoderLayer


@dataclass(frozen=True)
class LiLiCorrConfig:
    candidate_topk: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    factor_dim: int
    vector_eps: float
    logit_scale: float

    def resolve_hidden_size(self, *, model_hidden_size: int) -> int:
        return self.hidden_size or model_hidden_size

    @classmethod
    def from_dict(cls, config: dict) -> "LiLiCorrConfig":
        if not config.get("lilicorr_enabled", True):
            raise ValueError("LiLiCorrDraftModel requires lilicorr_enabled.")
        values = {}
        for name in cls.__dataclass_fields__:
            key = f"lilicorr_{name}"
            if key not in config:
                raise ValueError(f"Missing dflash_config.{key}.")
            cast = float if name in ("mlp_ratio", "vector_eps", "logit_scale") else int
            value = cast(config[key])
            if (
                not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "hidden_size")
            ):
                raise ValueError(f"Invalid dflash_config.{key}={value}.")
            values[name] = value
        result = cls(**values)
        if result.candidate_topk > 16 or result.candidate_topk & (
            result.candidate_topk - 1
        ):
            raise ValueError("LiLiCorr candidate_topk must be a power of two <= 16.")
        return result


class LiLiCorrRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)
        self._normalized_shape = (int(hidden_size),)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states, self._normalized_shape, self.weight, self.variance_epsilon
        )


class LiLiCorrLatticeAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible "
                f"by num_heads={num_heads}."
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)

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
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias.reshape(bsz, self.num_heads, seq_len, seq_len),
        )
        return self.out_proj(
            out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        )


class LiLiCorrLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, rms_norm_eps: float
    ) -> None:
        super().__init__()
        self.attn_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = LiLiCorrLatticeAttention(hidden_size, num_heads)
        self.mlp_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, hidden_size),
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
        config: LiLiCorrConfig,
    ) -> None:
        super().__init__()
        hidden_size = config.resolve_hidden_size(model_hidden_size=model_hidden_size)
        self.block_size = int(block_size)
        self.num_candidate_slots = self.block_size - 1
        self.candidate_topk = int(config.candidate_topk)
        self.hidden_size = hidden_size
        self.num_heads = int(config.num_heads)
        self.mlp_ratio = float(config.mlp_ratio)
        self.factor_dim = int(config.factor_dim)
        self.vector_eps = float(config.vector_eps)
        self.logit_scale = float(config.logit_scale)
        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(self.num_candidate_features),
            nn.Linear(self.num_candidate_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
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
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(int(config.num_layers))
            ]
        )
        self.output_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        self.factor_input_proj = nn.Linear(hidden_size * 3, hidden_size)
        self.out_head = nn.Linear(hidden_size, self.factor_dim)
        self.in_head = nn.Linear(hidden_size, self.factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, self.factor_dim)
        self._attn_bias: torch.Tensor | None = None
        self._fused_edge_weight: torch.Tensor | None = None
        self._fused_edge_bias: torch.Tensor | None = None
        self._factor_input_splits: tuple[torch.Tensor, ...] | None = None
        self._rank_frac_col: torch.Tensor | None = None
        self._is_top1_col: torch.Tensor | None = None

    @torch.no_grad()
    def materialize_inference_buffers(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        topk = self.candidate_topk
        self._attn_bias = self._build_attention_bias(device=device, dtype=dtype)
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
        weight = self.factor_input_proj.weight
        hdim = self.hidden_size
        self._factor_input_splits = (
            weight[:, :hdim].contiguous(),
            weight[:, hdim : 2 * hdim].contiguous(),
            weight[:, 2 * hdim :].contiguous(),
        )
        if topk > 1:
            rank_frac = torch.arange(topk, device=device, dtype=torch.float32).view(
                1, 1, 1, topk
            ) / float(topk - 1)
        else:
            rank_frac = torch.zeros(1, 1, 1, topk, device=device, dtype=torch.float32)
        is_top1 = torch.zeros(1, 1, 1, topk, device=device, dtype=torch.float32)
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
        rel = slot_ids.view(-1, 1) - slot_ids.view(1, -1)
        rel = rel.clamp(min=-(self.block_size - 1), max=self.block_size - 1)
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(
            dtype=bias.dtype
        ) * self.same_slot_bias.view(-1, 1, 1)
        return bias.to(device=device, dtype=dtype).contiguous()

    def _require_materialized(self) -> None:
        if self._attn_bias is None:
            raise RuntimeError(
                "Call materialize_inference_buffers() after loading the LiLiCorr head."
            )

    def _project_anchor(
        self, anchor_hidden: torch.Tensor, anchor_valid: torch.Tensor
    ) -> torch.Tensor:
        anchor = self.context_proj(anchor_hidden)
        return anchor * anchor_valid.unsqueeze(-1).to(anchor.dtype)

    def score(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
        already_projected: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_materialized()
        bsz, n_blocks, n_slots, topk = candidate_log_probs.shape
        if topk != self.candidate_topk:
            raise ValueError(
                f"LiLiCorr expects candidate_topk={self.candidate_topk}, got {topk}."
            )
        proj_dtype = self.pass_hidden_proj.weight.dtype
        if token_embeddings.dtype != proj_dtype:
            token_embeddings = token_embeddings.to(proj_dtype)
        if pass_hidden.dtype != proj_dtype:
            pass_hidden = pass_hidden.to(proj_dtype)
        token_states = (
            token_embeddings if already_projected else self.token_proj(token_embeddings)
        )
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
        hidden_states = hidden_states + self.slot_embedding
        hidden_states = hidden_states + self.rank_embedding
        hidden_states = hidden_states.reshape(
            bsz * n_blocks, n_slots * topk, self.hidden_size
        )
        anchor_state = self._project_anchor(anchor_hidden, anchor_valid)
        lattice = self._attn_bias.shape[-1]
        attention_bias = (
            self._attn_bias.unsqueeze(0)
            .expand(bsz * n_blocks, -1, -1, -1)
            .reshape(bsz * n_blocks * self.num_heads, lattice, lattice)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        hidden_states = self.output_norm(hidden_states).reshape(
            bsz, n_blocks, n_slots, topk, self.hidden_size
        )
        anchor_state = self.anchor_norm(anchor_state)
        w_self, w_anchor, w_cross = self._factor_input_splits
        anchor_row = anchor_state[:, :, None, None, :]
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
        start_scores = (anchor_out[:, :, None, :] * in_vec[:, :, 0, :, :]).sum(dim=-1)
        pair_scores = torch.matmul(
            out_vec[:, :, :-1], in_vec[:, :, 1:].transpose(-1, -2)
        )
        return (start_scores, pair_scores)

    def log_factors(
        self, start_scores: torch.Tensor, pair_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.logit_scale * start_scores.float(),
            self.logit_scale * pair_scores.float(),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> torch.Tensor:
        start, pairs = self.score(
            token_embeddings=token_embeddings.unsqueeze(1),
            candidate_log_probs=candidate_log_probs.unsqueeze(1),
            pass_hidden=pass_hidden.unsqueeze(1),
            anchor_hidden=anchor_hidden.unsqueeze(1),
            anchor_valid=anchor_valid.unsqueeze(1),
        )
        start, pairs = self.log_factors(start, pairs)
        # The shared walk starts at predecessor index zero; all first rows agree.
        first = start[:, 0, None, None, :].expand(-1, 1, self.candidate_topk, -1)
        return torch.cat((first, pairs[:, 0]), dim=1)


class LiLiCorrQwen3Model(DFlashQwen3Model):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        draft_config = config.dflash_config
        head_config = LiLiCorrConfig.from_dict(draft_config)
        block_size = int(
            draft_config.get("block_size", getattr(config, "block_size", 0))
        )
        if block_size != 1 + spec.num_speculative_tokens:
            raise ValueError(
                "LiLiCorr requires num_speculative_tokens = trained block_size - 1."
            )
        taps = int(draft_config.get("conv_kernel_size", 0))
        groups = int(draft_config.get("conv_group_size", 0))
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
                config=head_config,
            )


class LiLiCorrQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = LiLiCorrQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        spec = vllm_config.speculative_config
        assert spec is not None
        if spec.draft_model_config.quantization is not None:
            raise ValueError("Quantized LiLiCorr draft weights are not supported.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.draft_id_to_target_id is not None:
            raise ValueError("LiLiCorr candidates require the full target vocabulary.")
        # SGLang's exported head consumes raw target-head log-softmax features.
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
        expected = {
            name
            for name, _ in self.model.named_parameters()
            if name.startswith("lilicorr.")
            or ".attention_conv." in name
            or ".mlp_conv." in name
        }
        seen = set()

        def normalized_weights():
            for name, value in weights:
                name = name.removeprefix("model.")
                if (
                    name.startswith("lilicorr.")
                    or ".attention_conv." in name
                    or ".mlp_conv." in name
                ):
                    seen.add(name)
                yield name, value

        super().load_weights(normalized_weights())
        if seen != expected:
            raise ValueError(
                "LiLiCorr checkpoint coverage mismatch: "
                f"missing={sorted(expected - seen)}, "
                f"unexpected={sorted(seen - expected)}"
            )
        head = self.model.lilicorr
        parameter = next(head.parameters())
        head.materialize_inference_buffers(parameter.device, parameter.dtype)


EntryClass = LiLiCorrQwen3ForCausalLM
