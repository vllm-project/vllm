# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Laya typed-decision models (https://github.com/NandhaKishorM/laya).

A ModernBERT encoder, pre-norm bidirectional transformer layers conditioned
on the question type, and a scorer applied at each option's
``[MASK]`` marker. A request is one question rendered as::

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ...
    [SEP] <state> [SEP]

The ``token_classify`` output has one row per option marker: column 0 is the
option probability (or logit with ``use_activation=False``) and the remaining
columns are the action head's distribution, repeated on every row.
"""

import math
from collections.abc import Iterable, Set

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.models.modernbert import ModernBertModel
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import attn_type, default_pooling_type
from .utils import AutoWeightsLoader, WeightsMapper

QTYPES = ("choice", "score", "noul")
# Laya refuses calibration temperatures outside this range, see
# `laya.common.clamp_temperature`.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def _clamp_temperature(t: object) -> float:
    try:
        t = float(t)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def _temperature_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPES[qtype]}:{size}"


class LayaHeadSelfAttention(nn.Module):
    """`nn.MultiheadAttention` without a mask, over each whole sequence."""

    def __init__(self, hidden_size: int, num_heads: int, prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = EncoderOnlyAttention(
            num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.in_proj(hidden_states).split([self.hidden_size] * 3, dim=-1)
        return self.out_proj(self.attn(q, k, v))


class LayaHeadLayer(nn.Module):
    """`nn.TransformerEncoderLayer(norm_first=True, activation="relu")`."""

    def __init__(self, hidden_size: int, num_heads: int, prefix: str = ""):
        super().__init__()
        self.self_attn = LayaHeadSelfAttention(
            hidden_size, num_heads, prefix=f"{prefix}.self_attn"
        )
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.norm1(hidden_states))
        ffn = self.linear2(torch.relu(self.linear1(self.norm2(hidden_states))))
        return hidden_states + ffn


class LayaDecisionPooler(Pooler):
    """Scores each option marker and runs the action head on `[CLS]`."""

    def __init__(self, vllm_config: VllmConfig, hidden_size: int, n_act: int):
        super().__init__()
        laya_config = vllm_config.model_config.hf_config.laya_config
        self.mask_token_id: int = laya_config["mask_token_id"]
        self.qtype_token_ids: list[int] = laya_config["qtype_token_ids"]
        self.temperature = [
            _clamp_temperature(t) for t in laya_config.get("temperature", [1.0] * 3)
        ]
        self.temperature_by_options = {
            k: _clamp_temperature(v)
            for k, v in laya_config.get("temperature_by_options", {}).items()
        }

        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 256), nn.GELU(), nn.Linear(256, n_act)
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def _temperature(self, qtype: int, k: int) -> float:
        return self.temperature_by_options.get(
            _temperature_bucket(qtype, k), self.temperature[qtype]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        cursor = pooling_metadata.get_pooling_cursor()
        prompt_token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        device = hidden_states.device

        # Malformed prompts (e.g. profiling dummies) must not raise here:
        # they fall back to "choice" and give an empty output without markers.
        marker_idx, temperatures, num_markers = [], [], []
        for token_ids in prompt_token_ids:
            idx = (token_ids == self.mask_token_id).nonzero(as_tuple=True)[0]
            type_tok = int(token_ids[1]) if len(token_ids) > 1 else -1
            qtype = (
                self.qtype_token_ids.index(type_tok)
                if type_tok in self.qtype_token_ids
                else 0
            )
            marker_idx.append(idx)
            temperatures.append(self._temperature(qtype, len(idx)))
            num_markers.append(len(idx))

        # Gather each request's markers from the flattened batch into
        # a [num_reqs, max_markers] grid.
        offsets = torch.cumsum(cursor.num_scheduled_tokens_cpu, 0)
        offsets = torch.cat([offsets.new_zeros(1), offsets[:-1]])
        num_markers_cpu = torch.tensor(num_markers)
        mask_cpu = torch.arange(max(*num_markers, 1)) < num_markers_cpu[:, None]
        grid_idx_cpu = torch.zeros(mask_cpu.shape, dtype=torch.long)
        grid_idx_cpu[mask_cpu] = torch.cat([i + o for i, o in zip(marker_idx, offsets)])
        use_activation_cpu = torch.tensor(
            [p.use_activation is not False for p in pooling_metadata.pooling_params]
        )

        grid_idx = async_tensor_h2d(grid_idx_cpu, device)
        mask = async_tensor_h2d(mask_cpu, device)
        cls_idx = async_tensor_h2d(offsets, device)
        temperature = async_tensor_h2d(torch.tensor(temperatures), device)
        use_activation = async_tensor_h2d(use_activation_cpu, device)[:, None]

        logits = self.scorer(hidden_states[grid_idx]).squeeze(-1).float()
        logits = logits.masked_fill(~mask, -1e4)

        p = torch.softmax(logits, -1)
        k = mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = F.pad(p, (0, 1)).topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_input = torch.cat([hidden_states[cls_idx].float(), feats], -1)
        act_logits = self.act_head(act_input.to(hidden_states.dtype)).float()

        scores = torch.where(
            use_activation, torch.softmax(logits / temperature[:, None], -1), logits
        )
        act = torch.where(use_activation, torch.softmax(act_logits, -1), act_logits)
        act = act[:, None].expand(-1, scores.shape[1], -1)
        out = torch.cat([scores[..., None], act], -1)
        return [out[i, :n] for i, n in enumerate(num_markers)]


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class LayaForDecision(nn.Module):
    is_pooling_model = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Laya's Agent reads temperatures from the config, not this buffer.
            "temperature": None,
            "scorer.": "pooler.scorer.",
            "act_head.": "pooler.act_head.",
        },
        orig_to_new_suffix={
            ".in_proj_weight": ".in_proj.weight",
            ".in_proj_bias": ".in_proj.bias",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        laya_config = config.laya_config
        hidden_size = config.hidden_size
        num_heads = max(1, hidden_size // 64)

        self.encoder = ModernBertModel(vllm_config=vllm_config, prefix="encoder")
        self.type_emb = nn.Embedding(len(QTYPES), hidden_size)
        self.head = nn.Module()
        self.head.layers = nn.ModuleList(
            LayaHeadLayer(hidden_size, num_heads, prefix=f"head.layers.{i}")
            for i in range(laya_config["head_layers"])
        )
        self.register_buffer(
            "qtype_token_ids",
            torch.tensor(laya_config["qtype_token_ids"]),
            persistent=False,
        )
        n_act = len(laya_config.get("act_costs", {})) + 1
        self.pooler = LayaDecisionPooler(vllm_config, hidden_size, n_act)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_input_ids(input_ids)

    def _question_types(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-token question type, read from the token after each `[CLS]`."""
        idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        seq_start = torch.cummax(torch.where(positions == 0, idx, 0), 0).values
        type_tok = input_ids[(seq_start + 1).clamp(max=input_ids.shape[0] - 1)]
        return (type_tok[:, None] == self.qtype_token_ids).int().argmax(-1)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("Laya requires input_ids to read the question type")
        hidden_states = self.encoder(
            input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds
        )
        hidden_states = hidden_states + self.type_emb(
            self._question_types(input_ids, positions)
        )
        for layer in self.head.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
