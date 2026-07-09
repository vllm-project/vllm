# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained Domino/DFlash speculative decoding inference.

Minimal port of the Domino reference inference (jianuo-huang/Domino code/dflash.py)
adapted to load checkpoints trained with speculators and support both DFlash
(no GRU head) and Domino (with GRU correction head) projectors.

Key adaptations vs the reference:
- supports both projector_type="dflash" and "domino"
- embed_proj outputs draft_vocab_size, not target vocab
- per-position acceptance rate tracking
"""

import json
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Unpack

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time(device: torch.device | None = None) -> float:
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
    return time.perf_counter()


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _extract_context_feature(
    hidden_states: list[torch.Tensor], layer_ids: list[int] | None
) -> torch.Tensor:
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return torch.cat(selected, dim=-1)


# ---------------------------------------------------------------------------
# DFlash attention / decoder layer  (unchanged from Domino reference)
# ---------------------------------------------------------------------------


class _DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights


class _DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Draft model
# ---------------------------------------------------------------------------


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["_DFlashDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                _DFlashDecoderLayer(config, idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = config.dflash_config.get(
            "target_layer_ids",
            _build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = config.dflash_config.get("mask_token_id")
        self.projector_type = config.dflash_config.get("projector_type", "dflash")
        self.pure_draft_prefix_len = config.dflash_config.get(
            "pure_draft_prefix_len", 0
        )
        self.shift_label = config.dflash_config.get("shift_label", False)
        self.draft_vocab_size = config.dflash_config.get(
            "draft_vocab_size", config.vocab_size
        )

        self._is_domino = self.projector_type == "domino"
        if self._is_domino:
            self.emb_dim = config.dflash_config["emb_dim"]
            self.gru_hidden_dim = config.dflash_config["gru_hidden_dim"]
            self.prefix_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            in_dim = config.hidden_size + self.gru_hidden_dim
            # Output draft_vocab_size (paper x4.1.2: low-rank residual in draft vocab)
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, self.draft_vocab_size, bias=False),
            )

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    # ------------------------------------------------------------------
    #  Speculative generation loop
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def spec_generate(
        self,
        input_ids: torch.Tensor,
        target: nn.Module,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        stop_token_ids: list[int] | int | None = None,
        block_size: int | None = None,
        use_bias: bool = True,
        return_metrics: bool = False,
    ) -> torch.Tensor | SimpleNamespace:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "spec_generate supports input_ids with shape [1, seq_len]."
            )

        target_device = next(target.parameters()).device
        if target_device != self.device:
            raise ValueError("draft and target must be on the same device")

        input_ids = input_ids.to(self.device)
        block_size = int(block_size or self.block_size)
        mask_token_id = self.mask_token_id
        if mask_token_id is None:
            raise ValueError("dflash_config.mask_token_id is not set")

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        elif stop_token_ids is not None:
            stop_token_ids = list(stop_token_ids)

        # Build draft_id -> target_id mapping (for draft vocab -> target vocab)
        draft_vocab = self.draft_vocab_size
        target_vocab = target.config.vocab_size
        if draft_vocab != target_vocab:
            # DFlash checkpoints have a t2d/d2t mapping; default is identity first N tokens
            d2t = torch.arange(target_vocab, device=self.device)
            d2t[:draft_vocab] = torch.arange(draft_vocab, device=self.device)
        else:
            d2t = None

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + int(max_new_tokens)
        extra_buffer = block_size + 1 if self.shift_label else block_size

        output_ids = torch.full(
            (1, max_length + extra_buffer),
            mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(
            0
        )
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # --- Prefill ---
        prefill_start = _cuda_time(self.device)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=block_size > 1,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = _sample(
            output.logits, temperature
        )

        if block_size > 1:
            target_hidden = _extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )

        ttft = _cuda_time(self.device) - prefill_start

        # --- Decode loop ---
        decode_start = _cuda_time(self.device)
        start = num_input_tokens
        acceptance_lengths: list[int] = []
        draft_prefill = True
        prefix_len = int(self.pure_draft_prefix_len)
        pos_accept: list[list[int]] = [[] for _ in range(block_size)]

        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            k_draft = block_size if self.shift_label else block_size - 1
            verify_ids = torch.full(
                (1, k_draft + 1),
                mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            verify_ids[:, 0] = output_ids[:, start]
            verify_position_ids = position_ids[:, start : start + k_draft + 1]

            if block_size > 1:
                noise_embedding = target.model.embed_tokens(block_output_ids)
                parallel_hiddens = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )
                if not self.shift_label:
                    parallel_hiddens = parallel_hiddens[:, -block_size + 1 :, :]
                past_key_values_draft.crop(start)

                # Base logits from the verifier's frozen lm_head (matching training)
                base_logits = target.lm_head(
                    parallel_hiddens.to(target.lm_head.weight.dtype)
                ).float()

                # Pure-base prefix sampling
                if prefix_len > 0:
                    prefix_token_ids = _sample(base_logits[:, :prefix_len], temperature)
                    verify_ids[:, 1 : 1 + prefix_len] = prefix_token_ids

                if self._is_domino and use_bias:
                    realized_prefix_ids = verify_ids[:, : 1 + prefix_len]
                    realized_prefix_embeds = target.model.embed_tokens(
                        realized_prefix_ids
                    )
                    _, gru_hidden = self.prefix_gru(realized_prefix_embeds)

                    # Correction loop: start at _suffix_start (matching training)
                    # Training: _suffix_start = pure_prefix if shift_label else (1 + pure_prefix)
                    correction_start = (
                        prefix_len if self.shift_label else (prefix_len + 1)
                    )
                    for i in range(correction_start, k_draft):
                        z_i = parallel_hiddens[:, i : i + 1, :]
                        s_i = gru_hidden.transpose(0, 1)
                        bias = self.embed_proj(torch.cat([z_i, s_i], dim=-1))

                        # Correction in draft vocab space → map to target
                        if d2t is not None:
                            expanded_bias = bias.new_full(
                                (bias.shape[0], bias.shape[1], target_vocab),
                                0.0,
                            )
                            expanded_bias[:, :, d2t[:draft_vocab]] = bias
                            bias = expanded_bias

                        current_token_id = _sample(
                            base_logits[:, i : i + 1, :] + bias,
                            temperature,
                        )
                        verify_ids[:, i + 1 : i + 2] = current_token_id

                        if i + 1 < k_draft:
                            new_embed = target.model.embed_tokens(current_token_id)
                            _, gru_hidden = self.prefix_gru(new_embed, gru_hidden)
                else:
                    # Plain DFlash: sample remaining positions from base logits
                    for i in range(prefix_len, k_draft):
                        token = _sample(base_logits[:, i : i + 1, :], temperature)
                        verify_ids[:, i + 1 : i + 2] = token

                if draft_prefill:
                    draft_prefill = False
                    decode_start = _cuda_time(self.device)

            # --- Target verification ---
            output = target(
                verify_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=block_size > 1,
            )
            posterior = _sample(output.logits, temperature)

            acceptance_length = (
                (verify_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )

            # Track per-position acceptance — unconditional (fraction of all steps)
            matches = verify_ids[:, 1:] == posterior[:, :-1]  # [1, k_draft]
            cumprod_matches = matches.cumprod(dim=1)  # [1, k_draft]
            for p in range(k_draft):
                while len(pos_accept) <= p:
                    pos_accept.append([])
                pos_accept[p].extend([1 if cumprod_matches[0, p].item() else 0])

            # Track per-position acceptance — conditional on all priors accepted
            # matches = verify_ids[:, 1:] == posterior[:, :-1]  # [1, k_draft]
            # accepted_mask=matches.cumprod(dim=1).bool()  # [1, k_draft]
            # for p in range(k_draft):
            #     while len(pos_accept) <= p:
            #         pos_accept.append([])
            #     if p==0 or accepted_mask[0, p-1].item():
            #         pos_accept[p].extend([1 if matches[0, p].item() else 0])

            output_ids[:, start : start + int(acceptance_length) + 1] = verify_ids[
                :, : int(acceptance_length) + 1
            ]
            output_ids[:, start + int(acceptance_length) + 1] = posterior[
                :, int(acceptance_length)
            ]

            acceptance_lengths.append(int(acceptance_length) + 1)
            start += int(acceptance_length) + 1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = _extract_context_feature(
                    output.hidden_states,
                    self.target_layer_ids,
                )[:, : int(acceptance_length) + 1, :]

            if stop_token_ids is not None:
                stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                    break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != mask_token_id]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            idx = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(
                as_tuple=True
            )[0]
            if idx.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + idx[0].item() + 1]

        total_decode_time = _cuda_time(self.device) - decode_start

        if not return_metrics:
            return output_ids

        num_output_tokens = output_ids.shape[1] - num_input_tokens
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=ttft,
            time_per_output_token=total_decode_time / max(num_output_tokens, 1),
            total_decode_time=total_decode_time,
            acceptance_lengths=acceptance_lengths,
            pos_accept=pos_accept,
        )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_config_dict(checkpoint_dir: str) -> dict:
    with open(f"{checkpoint_dir}/config.json") as f:
        return json.load(f)


def _build_qwen3_config(cfg: dict) -> Qwen3Config:
    """Build a Qwen3Config from the speculators config.json."""
    tlc = cfg["transformer_layer_config"]
    qcfg = Qwen3Config(**tlc)
    # Overlay speculator-specific fields
    qcfg.num_target_layers = tlc.get("num_hidden_layers", 28)
    qcfg.block_size = cfg["block_size"]
    qcfg.dflash_config = {
        "target_layer_ids": cfg.get("aux_hidden_state_layer_ids"),
        "mask_token_id": cfg["mask_token_id"],
        "projector_type": cfg["projector_type"],
        "shift_label": cfg.get("shift_label", False),
        "pure_draft_prefix_len": cfg.get("pure_draft_prefix_len", 0),
        "emb_dim": cfg.get("emb_dim", 256),
        "gru_hidden_dim": cfg.get("gru_hidden_dim", 1024),
        "draft_vocab_size": cfg["draft_vocab_size"],
        "block_size": cfg["block_size"],
    }
    return qcfg


def _load_weights_into(model: DFlashDraftModel, checkpoint_dir: str, is_domino: bool):
    """Load safetensors weights into the model, with key remapping."""
    state = load_file(f"{checkpoint_dir}/model.safetensors")
    model_state = model.state_dict()
    loaded = 0

    for key, value in state.items():
        target_key = key

        # Map Domino head weights
        if is_domino:
            if key.startswith("domino_head.prefix_gru.") or key.startswith(
                "domino_head.embed_proj."
            ):
                target_key = key.replace("domino_head.", "")

        # Map backbone weights
        if target_key.startswith("prefix_gru.") or target_key.startswith("embed_proj."):
            pass  # handled above
        elif target_key.startswith("model."):
            target_key = target_key[6:]  # strip "model." prefix
        elif target_key in ("fc.weight", "hidden_norm.weight", "norm.weight"):
            pass  # keep as-is
        elif target_key == "lm_head.weight":
            continue  # not used; base logits come from target.lm_head
        elif "layers." in target_key or "rotary_emb." in target_key:
            pass
        else:
            continue  # skip metadata keys

        if target_key in model_state:
            param = model_state[target_key]
            if param.shape == value.shape:
                param.copy_(value)
                loaded += 1
            else:
                print(
                    f"  [WARN] shape mismatch: {target_key} {param.shape} vs {value.shape}"
                )
        else:
            if is_domino and (
                target_key.startswith("prefix_gru.")
                or target_key.startswith("embed_proj.")
            ):
                print(f"  [SKIP] {target_key} not found in model state")

    print(f"  Loaded {loaded}/{len(state)} weights")


def load_draft_model(checkpoint_dir: str, device: torch.device) -> DFlashDraftModel:
    """Load a DFlash/Domino draft model from a speculators checkpoint."""
    cfg = _load_config_dict(checkpoint_dir)
    qcfg = _build_qwen3_config(cfg)
    model = DFlashDraftModel(qcfg)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    _load_weights_into(model, checkpoint_dir, model._is_domino)
    print(
        f"  Draft model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    return model
