# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


# PROBE: dump sub-op outputs at the position(s) of token 2701 in the current
# forward call, tagged by a monotonic trigger index, for cross-config and
# cross-call comparison. Filtering is gated by qwen2 module-level state set
# in Qwen2Model.forward, which only marks a forward call as "interesting"
# when input_ids actually contains 2701. This eliminates noise from warmup,
# decode-only forwards, and forwards that prefill other prompts.
#
# Post-process: the same trigger_idx in baseline vs failing config dumps the
# same logical prefill (greedy + same prompt order ensures determinism in
# scheduling). For each (trigger_idx, op, pos) tuple, compare first8 across
# configs to find the first op whose output differs.
def _dump_subop_out(name: str, output):
    import os as _os

    if _os.environ.get("VLLM_DET_CHECK", "") != "1":
        return
    from . import qwen2 as _qwen2

    positions = _qwen2._probe_target_positions
    if not positions:
        return
    t = output[0] if isinstance(output, tuple) else output
    if not isinstance(t, torch.Tensor):
        return
    if t.ndim < 2:
        return
    trigger = _qwen2._probe_trigger_idx
    M = _qwen2._probe_batch_M
    import sys as _sys

    for pos in positions[:2]:
        if pos >= t.shape[0]:
            continue
        flat = t[pos].detach().reshape(-1)[:8].cpu().float().tolist()
        print(
            f"[SUBOP_OUT] trigger={trigger} M={M} op={name} pos={pos} "
            f"shape={list(t.shape)} first8={flat}",
            file=_sys.stderr,
            flush=True,
        )


# PROBE: dump qkv_proj weight checksum once per trigger, only from layer 0,
# to rule out weight-drift as an explanation for output divergence. We log
# (a) sum of weight[0, :8], (b) sum of full weight tensor, both as float64
# to avoid the fp32 sum precision issue. If these match across configs but
# qkv_proj output still differs, the weight is not the cause.
_weight_dumped_for_trigger: int = -1


def _dump_weight_once_per_trigger(weight: torch.Tensor):
    import os as _os

    if _os.environ.get("VLLM_DET_CHECK", "") != "1":
        return
    from . import qwen2 as _qwen2

    if not _qwen2._probe_target_positions:
        return
    global _weight_dumped_for_trigger
    trigger = _qwen2._probe_trigger_idx
    if trigger == _weight_dumped_for_trigger:
        return
    _weight_dumped_for_trigger = trigger
    try:
        w0 = weight[0, :8].detach().cpu().float().tolist()
        wsum = float(weight.detach().double().sum().item())
        wshape = list(weight.shape)
    except Exception:
        return
    import sys as _sys

    print(
        f"[SUBOP_WEIGHT] trigger={trigger} op=qkv_proj shape={wshape} "
        f"row0_first8={w0} fullsum={wsum:.18g}",
        file=_sys.stderr,
        flush=True,
    )


# Wrapper that calls fn once and dumps its output. Replaces the old
# within-call _det_check_call which we showed only catches within-call
# non-determinism — not the across-generate variety we need.
def _det_check_call(name: str, fn, *args):
    out = fn(*args)
    _dump_subop_out(name, out)
    return out


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # PROBE: only run det check on layer 0 (the first divergence point).
        _det = (
            getattr(self, "_probe_layer_idx", -1) == 0
            and __import__("os").environ.get("VLLM_DET_CHECK", "") == "1"
        )
        if _det:
            # PROBE: weight checksum once per trigger to rule out weight drift.
            _dump_weight_once_per_trigger(self.qkv_proj.weight)
            qkv, _ = _det_check_call("qkv_proj", self.qkv_proj, hidden_states)
            # PROBE: within-call re-run with the SAME input + weight. If qkv ==
            # qkv_recheck bit-for-bit, cuBLAS is deterministic for this shape
            # in this forward — and any cross-config divergence is purely a
            # function of the shape (M-dim) or pre-forward GPU state. If they
            # differ, cuBLAS handle/workspace state is leaking across calls
            # within the same forward, and shape-only theories are wrong.
            qkv_recheck, _ = self.qkv_proj(hidden_states)
            _dump_subop_out("qkv_proj_recheck", qkv_recheck)
            # PROBE: float64 reference. F.linear(input.double(), weight.double())
            # gives an effectively-exact result; the config whose qkv is closer
            # to this is the "correct" one. Both being equally far means cuBLAS
            # is picking different but equally-valid algos.
            try:
                import torch.nn.functional as _F

                qkv_ref = _F.linear(
                    hidden_states.detach().double(),
                    self.qkv_proj.weight.detach().double(),
                )
                _dump_subop_out("qkv_proj_ref_fp64", qkv_ref.float())
            except Exception:
                pass
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        if _det:
            q_by_head = _det_check_call("q_norm", self.q_norm, q_by_head)
        else:
            q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        if _det:
            k_by_head = _det_check_call("k_norm", self.k_norm, k_by_head)
        else:
            k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        if _det:
            q, k = _det_check_call("rotary_emb", self.rotary_emb, positions, q, k)
        else:
            q, k = self.rotary_emb(positions, q, k)
        # PROBE: also dump q/k AFTER rotary (the input to attention) and
        # attn_output (the output of attention) for cross-pass comparison.
        # self.attn mutates KV cache so we don't re-call it, but we still
        # dump its output to see if it differs across passes.
        if _det:
            _dump_subop_out("q_after_rotary", q)
            _dump_subop_out("k_after_rotary", k)
            _dump_subop_out("v_input", v)
        attn_output = self.attn(q, k, v)
        if _det:
            _dump_subop_out("attn_output", attn_output)
            output, _ = _det_check_call("o_proj", self.o_proj, attn_output)
        else:
            output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # PROBE: layer index for selective det check (only layer 0)
        self._probe_layer_idx = extract_layer_index(prefix)
        self.self_attn._probe_layer_idx = self._probe_layer_idx

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import os as _os

        _det = (
            self._probe_layer_idx == 0 and _os.environ.get("VLLM_DET_CHECK", "") == "1"
        )
        # Self Attention
        if residual is None:
            residual = hidden_states
            if _det:
                hidden_states = _det_check_call(
                    "input_layernorm", self.input_layernorm, hidden_states
                )
            else:
                hidden_states = self.input_layernorm(hidden_states)
        else:
            if _det:
                hidden_states, residual = _det_check_call(
                    "input_layernorm_w_residual",
                    self.input_layernorm,
                    hidden_states,
                    residual,
                )
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        if _det:
            hidden_states, residual = _det_check_call(
                "post_attention_layernorm",
                self.post_attention_layernorm,
                hidden_states,
                residual,
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        if _det:
            hidden_states = _det_check_call("mlp", self.mlp, hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


class Qwen3ForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        # PROBE: scan ALL rows of logits, computing logprob[2701] = logit - lse
        # at each row. Find the row(s) where logprob matches the failure
        # signature (-10.5131..., the value at the failing prompt position)
        # and dump their rank and near-tie candidates.
        import os as _os

        if (
            _os.environ.get("VLLM_DEBUG_DUMP_HS", "") == "1"
            and logits is not None
            and logits.ndim == 2
            and logits.shape[0] >= 2
            and logits.shape[1] > 2701
        ):
            import sys

            logits_gpu = logits.detach()
            row_max = logits_gpu.max(dim=1).values  # [num_rows]
            row_lse = (logits_gpu - row_max.unsqueeze(1)).exp().sum(
                dim=1
            ).log() + row_max
            tok2701_per_row = logits_gpu[:, 2701]  # [num_rows]
            logprob_2701_per_row = tok2701_per_row - row_lse  # [num_rows]

            # Find rows whose logprob[2701] is near the failure signature
            target = -10.5131
            near_target_mask = (logprob_2701_per_row - target).abs() < 0.001
            near_rows = near_target_mask.nonzero(as_tuple=False).flatten().tolist()

            for row_idx in near_rows[:5]:
                row = logits_gpu[row_idx]
                tok2701_val = float(row[2701].item())
                logprob_val = float(logprob_2701_per_row[row_idx].item())
                lse_val = float(row_lse[row_idx].item())
                rank_count = int((row >= row[2701]).sum().item())
                near_mask = (row - row[2701]).abs() < 1e-6
                near_idx = near_mask.nonzero(as_tuple=False).flatten().tolist()
                near_vals = [(int(i), float(row[i].item())) for i in near_idx[:8]]
                print(
                    f"[VLLM_LOGITS] shape={list(logits.shape)} row={row_idx} "
                    f"tok2701={tok2701_val:.18g} lse={lse_val:.18g} "
                    f"logprob={logprob_val:.18g} rank_count={rank_count} "
                    f"near_count={len(near_idx)} near={near_vals}",
                    file=sys.stderr,
                    flush=True,
                )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
