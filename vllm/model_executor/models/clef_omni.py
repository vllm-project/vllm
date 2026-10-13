# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clef-Omni (https://huggingface.co/Cloudflare/clef-omni).

The Qwen3-Omni thinker with a joint schema head that scores every option of
every question in one forward pass. The head reads the final hidden states of
the whole prompt, so it runs as a ``token_classify`` pooler.

Each request carries its schema in ``PoolingParams.extra_kwargs`` under
``"joint_schema"``:

- ``question_types``: 0 (noul), 1 (choice) or 2 (score) per question.
- ``question_spans``: the ``[start, end)`` of each question's instructions.
- ``option_spans``: the ``[start, end)`` of each option, question by question.
- ``option_counts``: the number of options of each question.

Spans count from the end of the prompt (``-len(prompt) <= start < end <= 0``),
so they hold after multimodal placeholders are expanded. The output has one
value per option, in ``option_spans`` order: its probability within its
question, or its logit with ``use_activation=False``.
"""

import math
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type
from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)

JOINT_HEAD_WEIGHTS = "joint_head.safetensors"
NUM_QUESTION_TYPES = 3
MAX_LOGIT_SCALE = math.log(100.0)


def segment_mean(
    values: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    sums = values.new_zeros(lengths.shape[0], values.shape[-1], dtype=torch.float32)
    sums.index_add_(0, segment_ids, values.float())
    return (sums / lengths[:, None]).to(values.dtype)


@dataclass
class SchemaIndices:
    """Where one prompt's questions and options are, on the model's device."""

    positions: torch.Tensor
    """Every token of every span, questions first."""
    span_ids: torch.Tensor
    """The span of each of ``positions``."""
    span_lengths: torch.Tensor
    """``[Q + O]``."""
    question_types: torch.Tensor
    """``[Q]``."""
    option_question: torch.Tensor
    """The question of each option, ``[O]``."""
    option_slots: torch.Tensor
    """Each option's flat index in a ``[Q, max_options]`` grid, ``[O]``."""
    option_mask: torch.Tensor
    """``[Q, max_options]``, true at the slots of options."""


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        # The Identity takes the place of the checkpoint's dropout.
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(feedforward, width),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Separate key and value tensors, like the reference, keep attention's
        # unpacked input projections.
        routed, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(NUM_QUESTION_TYPES, width)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(width, 1),
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        lexical: torch.Tensor,
        schema: SchemaIndices,
    ) -> torch.Tensor:
        """The ``[O]`` option logits of one prompt, from its ``[L, hidden]``
        final hidden states and each option's ``[O, hidden]`` mean output
        embedding."""
        num_questions = schema.question_types.shape[0]
        option_question = schema.option_question
        hidden_states = self.hidden_norm(hidden_states)
        memory = self.memory_projection(hidden_states).unsqueeze(0)
        global_vector = hidden_states[-1]
        means = segment_mean(
            hidden_states[schema.positions], schema.span_ids, schema.span_lengths
        )
        question_vectors, contexts = means[:num_questions], means[num_questions:]

        options = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors)[option_question]
        ).unsqueeze(0)
        for layer in self.evidence_layers:
            options = layer(options, memory)
        options = options[0]

        # Each field summarizes its options, weighted by their match to it.
        base_fields = self.question_projection(question_vectors)
        width = options.shape[-1]
        grid = options.new_zeros(schema.option_mask.numel(), width)
        grid[schema.option_slots] = options
        grid = grid.view(*schema.option_mask.shape, width)
        routing = torch.einsum("qow,qw->qo", grid, base_fields) / math.sqrt(width)
        routing = routing.masked_fill(~schema.option_mask, float("-inf"))
        summaries = torch.einsum("qo,qow->qw", routing.softmax(dim=-1), grid)
        fields = (
            base_fields
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector).unsqueeze(0)
            + self.type_embedding(schema.question_types)
        ).unsqueeze(0)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[option_question]

        anchors = F.normalize(question_vectors + global_vector, dim=-1)
        prior_scale = self.prior_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        prior = prior_scale * (
            F.normalize(lexical, dim=-1) * anchors[option_question]
        ).sum(dim=-1)
        options = self.option_norm(options)
        cosine = F.cosine_similarity(fields, options, dim=-1)
        features = torch.cat(
            [fields, options, fields * options, torch.abs(fields - options)], dim=-1
        )
        residual = self.residual_scorer(features).squeeze(-1)
        joint_scale = self.joint_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        joint = joint_scale * cosine + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


def parse_joint_schema(
    extra_kwargs: dict[str, Any] | None, prompt_len: int
) -> tuple[list[int], list[tuple[int, int]], list[int]] | None:
    """The question types, the absolute spans of the questions then of the
    options, and the option counts. None if the schema is missing or invalid."""
    schema = (extra_kwargs or {}).get("joint_schema")
    if not isinstance(schema, dict):
        return None
    try:
        types = [int(t) for t in schema["question_types"]]
        counts = [int(c) for c in schema["option_counts"]]
        question_spans = [
            (int(start) + prompt_len, int(end) + prompt_len)
            for start, end in schema["question_spans"]
        ]
        option_spans = [
            (int(start) + prompt_len, int(end) + prompt_len)
            for start, end in schema["option_spans"]
        ]
    except (KeyError, TypeError, ValueError):
        return None
    spans = question_spans + option_spans
    if (
        not types
        or len(counts) != len(types)
        or len(question_spans) != len(types)
        or len(option_spans) != sum(counts)
        or any(not 0 <= t < NUM_QUESTION_TYPES for t in types)
        or any(c < 1 for c in counts)
        or any(not 0 <= start < end <= prompt_len for start, end in spans)
    ):
        return None
    return types, spans, counts


class JointSchemaPooler(Pooler):
    """Runs the joint schema head on each finished prompt.

    It never raises, because it runs in the engine core: a prompt without a
    valid schema, such as a profiling dummy, gets an empty output.
    """

    def __init__(
        self,
        head: JointSchemaHead,
        embed_lexical: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.pooling = AllPool()
        self.head = head
        self.embed_lexical = embed_lexical

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        outputs: list[torch.Tensor | None] = []
        for hidden, token_ids, params in zip(
            pooled, prompt_token_ids, pooling_metadata.pooling_params
        ):
            if hidden is None:
                outputs.append(None)
                continue
            parsed = parse_joint_schema(params.extra_kwargs, hidden.shape[0])
            if parsed is None:
                outputs.append(hidden.new_empty(0, dtype=torch.float32))
                continue
            schema, lexical_ids, lexical_span_ids = self._indices(
                *parsed, hidden.device
            )
            lexical = segment_mean(
                self.embed_lexical(
                    async_tensor_h2d(token_ids[lexical_ids], hidden.device)
                ),
                lexical_span_ids,
                schema.span_lengths[len(parsed[0]) :],
            )
            logits = self.head(hidden, lexical, schema).float()
            if params.use_activation is not False:
                mask = schema.option_mask
                grid = logits.new_full((mask.numel(),), float("-inf"))
                grid[schema.option_slots] = logits
                probs = grid.view(mask.shape).softmax(dim=-1)
                logits = probs.view(-1)[schema.option_slots]
            outputs.append(logits)
        return outputs

    @staticmethod
    def _indices(
        types: list[int],
        spans: list[tuple[int, int]],
        counts: list[int],
        device: torch.device,
    ) -> tuple[SchemaIndices, torch.Tensor, torch.Tensor]:
        """The schema's indices, plus the option tokens' positions (on the CPU)
        and their option spans, for the lexical embeddings."""
        num_questions = len(types)
        lengths = torch.tensor([end - start for start, end in spans])
        span_ids = torch.repeat_interleave(torch.arange(len(spans)), lengths)
        positions = torch.cat([torch.arange(start, end) for start, end in spans])
        num_question_tokens = int(lengths[:num_questions].sum())

        counts_cpu = torch.tensor(counts)
        max_options = max(counts)
        option_question = torch.repeat_interleave(
            torch.arange(num_questions), counts_cpu
        )
        option_slots = torch.cat(
            [torch.arange(count) + q * max_options for q, count in enumerate(counts)]
        )
        schema = SchemaIndices(
            positions=async_tensor_h2d(positions, device),
            span_ids=async_tensor_h2d(span_ids, device),
            span_lengths=async_tensor_h2d(lengths, device),
            question_types=async_tensor_h2d(types, device, dtype=torch.long),
            option_question=async_tensor_h2d(option_question, device),
            option_slots=async_tensor_h2d(option_slots, device),
            option_mask=async_tensor_h2d(
                torch.arange(max_options) < counts_cpu[:, None], device
            ),
        )
        lexical_span_ids = async_tensor_h2d(
            span_ids[num_question_tokens:] - num_questions, device
        )
        return schema, positions[num_question_tokens:], lexical_span_ids


MAX_FRAME_PIXELS = 256 * 32 * 32


def cap_frame_pixels(frames: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """The reference reads each video frame at at most 256 * 32 * 32 pixels,
    in even sizes. Frames are NHWC uint8, numpy or tensor; tensors stay on
    their device (a CUDA tensor from torchcodec must not round-trip)."""
    height, width = frames.shape[1:3]
    if height * width <= MAX_FRAME_PIXELS:
        return frames
    scale = math.sqrt(MAX_FRAME_PIXELS / (width * height))
    new_width = max(2, round(width * scale / 2) * 2)
    new_height = max(2, round(height * scale / 2) * 2)
    is_tensor = isinstance(frames, torch.Tensor)
    tensor = frames if is_tensor else torch.from_numpy(frames)
    resized = F.interpolate(
        tensor.permute(0, 3, 1, 2).float(),
        size=(new_height, new_width),
        mode="bilinear",
    )
    if tensor.dtype == torch.uint8:
        resized = resized.round()
    resized = resized.to(tensor.dtype).permute(0, 2, 3, 1).contiguous()
    return resized if is_tensor else resized.numpy()


def _cap_video(item: Any) -> Any:
    if isinstance(item, tuple):
        frames, metadata = item
        return cap_frame_pixels(frames), metadata
    return cap_frame_pixels(item)


class ClefOmniMultiModalProcessor(Qwen3OmniMoeThinkerMultiModalProcessor):
    """Caps video frames at the reference's per-frame pixel count."""

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if videos := hf_data.get("videos"):
            assert isinstance(videos, list)
            hf_data = {**hf_data, "videos": [_cap_video(v) for v in videos]}
        return super()._call_hf_processor(hf_data, hf_kwargs)


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    ClefOmniMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class ClefOmniForJointSchema(Qwen3OmniMoeThinkerForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        head_config = model_config.hf_config.joint_head_config
        self.pooler = JointSchemaPooler(
            JointSchemaHead(**head_config), self._embed_lexical
        )
        # The head's weights ship as their own safetensors file.
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="pooler.head.",
                allow_patterns_overrides=[JOINT_HEAD_WEIGHTS],
            )
        ]

    def _embed_lexical(self, token_ids: torch.Tensor) -> torch.Tensor:
        # The head embeds options with the output embeddings. ParallelLMHead
        # forbids lookups, so use the vocab-parallel embedding's.
        return VocabParallelEmbedding.forward(self.language_model.lm_head, token_ids)
