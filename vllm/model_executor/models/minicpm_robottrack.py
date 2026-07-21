# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-RobotTrack model compatible with HuggingFace weights.

MiniCPM-RobotTrack is a vision-language-action policy: a bare MiniCPM4-0.5B
decoder backbone plus an input adapter (vision projector + temporal markers +
learnable control query) and a funnel trajectory head that regresses eight
``[x, y, yaw]`` waypoints. It is non-generative (a single causal forward whose
last token drives the head), so it is served as a vLLM pooling model that
advertises the ``"embed"`` task and returns a flat 24-dim vector per request
(reshape to ``[8, 3]`` on the client).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import LastPool, SequencePooler
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import default_pooling_type
from .minicpm import MiniCPMModel
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class VisionProjector(nn.Module):
    """Map fused DINOv3+SigLIP features into the MiniCPM hidden space."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TemporalMarkerEncoder(nn.Module):
    """Build one marker token per represented frame (time+stream+camera)."""

    def __init__(self, hidden_dim: int, max_time_steps: int) -> None:
        super().__init__()
        self.time_embedding = nn.Embedding(max_time_steps, hidden_dim)
        self.stream_embedding = nn.Embedding(2, hidden_dim)
        self.camera_embedding = nn.Embedding(1, hidden_dim)

    def forward(
        self, time_step: int, stream_id: int, device: torch.device
    ) -> torch.Tensor:
        time = torch.tensor([time_step], dtype=torch.long, device=device)
        stream = torch.tensor([stream_id], dtype=torch.long, device=device)
        camera = torch.zeros(1, dtype=torch.long, device=device)
        return (
            self.time_embedding(time)
            + self.stream_embedding(stream)
            + self.camera_embedding(camera)
        ).squeeze(0)


class FunnelTrajectoryHead(nn.Module):
    """Six-layer funnel MLP predicting a fixed waypoint trajectory."""

    def __init__(
        self,
        hidden_dim: int,
        num_waypoints: int,
        action_dim: int,
        dropout: float,
        use_tanh: bool,
    ) -> None:
        super().__init__()
        output_dim = num_waypoints * action_dim
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim
        self.use_tanh = use_tanh
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, control_state: torch.Tensor) -> torch.Tensor:
        trajectory = self.layers(control_state)
        if self.use_tanh:
            trajectory = torch.tanh(trajectory)
        return trajectory.view(-1, self.num_waypoints, self.action_dim)


def _robottrack_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    coarse_lengths = hf_inputs.get("coarse_lengths", torch.empty(0, dtype=torch.long))
    fine_lengths = hf_inputs.get("fine_lengths", torch.empty(0, dtype=torch.long))
    return dict(
        coarse_tokens=MultiModalFieldConfig.flat_from_sizes("image", coarse_lengths),
        coarse_time_indices=MultiModalFieldConfig.flat_from_sizes(
            "image", coarse_lengths
        ),
        fine_tokens=MultiModalFieldConfig.flat_from_sizes("image", fine_lengths),
        fine_time_indices=MultiModalFieldConfig.flat_from_sizes("image", fine_lengths),
        coarse_lengths=MultiModalFieldConfig.batched("image"),
        fine_lengths=MultiModalFieldConfig.batched("image"),
    )


class MiniCPMRobotTrackImageItems(DictEmbeddingItems):
    """One item = the full visual bundle (coarse/fine tokens + time indices)."""

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Any,
    ) -> None:
        super().__init__(
            data,
            modality="image",
            required_fields={
                "coarse_tokens",
                "coarse_time_indices",
                "fine_tokens",
                "fine_time_indices",
            },
            fields_factory=fields_factory,
        )


class MiniCPMRobotTrackDataParser(MultiModalDataParser):
    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if not isinstance(data, dict):
            raise ValueError(
                "MiniCPM-RobotTrack expects precomputed visual features passed "
                "as a dict of tensors under the 'image' modality."
            )
        data = _with_visual_lengths(data)
        return MiniCPMRobotTrackImageItems(
            data,
            fields_factory=_robottrack_field_config,
        )


def _with_visual_lengths(
    data: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Attach per-item token counts so the flat fields can be sliced back."""
    out = dict(data)
    coarse = torch.as_tensor(out["coarse_tokens"])
    fine = torch.as_tensor(out["fine_tokens"])
    out["coarse_tokens"] = coarse
    out["fine_tokens"] = fine
    out["coarse_time_indices"] = torch.as_tensor(
        out["coarse_time_indices"], dtype=torch.long
    )
    out["fine_time_indices"] = torch.as_tensor(
        out["fine_time_indices"], dtype=torch.long
    )
    out["coarse_lengths"] = torch.tensor([coarse.shape[0]], dtype=torch.long)
    out["fine_lengths"] = torch.tensor([fine.shape[0]], dtype=torch.long)
    return out


def _count_marker_runs(time_indices: torch.Tensor) -> int:
    """Number of maximal equal-value runs (one temporal marker per run)."""
    values = time_indices.tolist()
    if not values:
        return 0
    runs = 1
    for prev, cur in zip(values, values[1:]):
        if cur != prev:
            runs += 1
    return runs


class MiniCPMRobotTrackProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        return MiniCPMRobotTrackDataParser()

    def get_image_token_id(self) -> int:
        # A reserved in-vocab token id (last vocab slot, a special token that
        # never appears in a natural-language instruction) marks the appended
        # visual bundle; those positions are overwritten by the visual embeds.
        return self.get_hf_config().backbone_config.vocab_size - 1

    def get_num_image_tokens(
        self,
        coarse_time_indices: torch.Tensor,
        fine_time_indices: torch.Tensor,
    ) -> int:
        coarse = int(coarse_time_indices.shape[0]) + _count_marker_runs(
            coarse_time_indices
        )
        fine = int(fine_time_indices.shape[0]) + _count_marker_runs(fine_time_indices)
        return coarse + fine + 1  # +1 for the control query

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Worst case: every visual token sits in its own frame -> a marker each.
        coarse = hf_config.history_frames * hf_config.coarse_tokens_per_frame
        fine = hf_config.fine_tokens_current_frame
        return {"image": 2 * coarse + 2 * fine + 1}


class MiniCPMRobotTrackDummyInputsBuilder(
    BaseDummyInputsBuilder[MiniCPMRobotTrackProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {}
        hf_config = self.info.get_hf_config()
        dim = hf_config.vision_feature_dim
        frames = hf_config.history_frames
        coarse_per = hf_config.coarse_tokens_per_frame
        fine = hf_config.fine_tokens_current_frame
        num_coarse = frames * coarse_per
        coarse_tokens = torch.zeros(num_coarse, dim)
        coarse_time_indices = torch.arange(frames).repeat_interleave(coarse_per)
        fine_tokens = torch.zeros(fine, dim)
        fine_time_indices = torch.full((fine,), frames, dtype=torch.long)
        return {
            "image": {
                "coarse_tokens": coarse_tokens,
                "coarse_time_indices": coarse_time_indices,
                "fine_tokens": fine_tokens,
                "fine_time_indices": fine_time_indices,
            }
        }


class MiniCPMRobotTrackMultiModalProcessor(
    BaseMultiModalProcessor[MiniCPMRobotTrackProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        return self.info.get_data_parser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # There is no HF multimodal processor: only tokenize the instruction.
        # The visual bundle is appended by `_get_prompt_updates` (insertion at
        # the end), matching HF's [text, history, current, control] order.
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        return BatchFeature(
            {"input_ids": [input_ids]},
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _robottrack_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: Any,
    ) -> Sequence[PromptUpdate]:
        image_items = mm_items.get_items("image", MiniCPMRobotTrackImageItems)
        image_token_id = self.info.get_image_token_id()

        def get_insertion(item_idx: int) -> list[int]:
            item = image_items.get(item_idx)
            num_tokens = self.info.get_num_image_tokens(
                item["coarse_time_indices"],
                item["fine_time_indices"],
            )
            return [image_token_id] * num_tokens

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.end(),
                insertion=get_insertion,
            )
        ]


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMRobotTrackMultiModalProcessor,
    info=MiniCPMRobotTrackProcessingInfo,
    dummy_inputs=MiniCPMRobotTrackDummyInputsBuilder,
)
class MiniCPMRobotTrackModel(nn.Module, SupportsMultiModal):
    is_pooling_model = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"backbone.": "model."})

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        backbone_config = config.backbone_config
        hidden_dim = backbone_config.hidden_size

        with self._mark_language_model(vllm_config):
            self.model = MiniCPMModel(
                vllm_config=vllm_config.with_hf_config(backbone_config),
                prefix=maybe_prefix(prefix, "model"),
            )

        self.vision_projector = VisionProjector(config.vision_feature_dim, hidden_dim)
        self.temporal_markers = TemporalMarkerEncoder(hidden_dim, config.max_time_steps)
        self.control_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.trajectory_head = FunnelTrajectoryHead(
            hidden_dim=hidden_dim,
            num_waypoints=config.num_waypoints,
            action_dim=config.action_dim,
            dropout=config.trajectory_dropout,
            use_tanh=config.use_tanh_actions,
        )

        # output_scale is a fixed reshaping constant; recompute it rather than
        # loading it from the checkpoint so it never depends on stored dtype.
        output_scale = torch.ones(1, 1, config.action_dim)
        output_scale[..., :2] = config.xy_scale
        self.register_buffer("output_scale", output_scale, persistent=False)

        self.pooler = DispatchPooler(
            {"embed": SequencePooler(pooling=LastPool(), head=self._pool_trajectory)}
        )

    def _pool_trajectory(
        self,
        pooled_data: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        head_dtype = self.trajectory_head.layers[-1].weight.dtype
        trajectory = self.trajectory_head(pooled_data.to(head_dtype))
        trajectory = trajectory * self.output_scale.to(trajectory.dtype)
        return trajectory.flatten(1)

    def _embed_text_input_ids(
        self,
        input_ids: torch.Tensor,
        embed_input_ids: Any,
        *,
        is_multimodal: torch.Tensor | None,
    ) -> torch.Tensor:
        # HF RobotTrack feeds RAW token embeddings (no scale_emb) as
        # inputs_embeds, so bypass MiniCPMModel.embed_input_ids (which multiplies
        # by scale_emb) and use the plain embedding table instead.
        return super()._embed_text_input_ids(
            input_ids,
            self.model.embed_tokens,
            is_multimodal=is_multimodal,
        )

    def _insert_temporal_markers(
        self,
        tokens: torch.Tensor,
        time_indices: torch.Tensor,
        stream_id: int,
    ) -> torch.Tensor:
        if tokens.shape[0] == 0:
            return tokens
        device = tokens.device
        time_row = time_indices.tolist()
        pieces: list[torch.Tensor] = []
        start = 0
        while start < len(time_row):
            time_step = int(time_row[start])
            end = start + 1
            while end < len(time_row) and int(time_row[end]) == time_step:
                end += 1
            marker = self.temporal_markers(time_step, stream_id, device)
            pieces.append(marker.unsqueeze(0).to(tokens.dtype))
            pieces.append(tokens[start:end])
            start = end
        return torch.cat(pieces, dim=0)

    def _embed_visual_bundle(
        self,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
    ) -> torch.Tensor:
        device = self.control_query.device
        proj_dtype = self.vision_projector.layers[1].weight.dtype
        coarse_tokens = coarse_tokens.to(device=device, dtype=proj_dtype)
        fine_tokens = fine_tokens.to(device=device, dtype=proj_dtype)
        history = self.vision_projector(coarse_tokens)
        current = self.vision_projector(fine_tokens)
        history = self._insert_temporal_markers(
            history, coarse_time_indices.to(device), stream_id=0
        )
        current = self._insert_temporal_markers(
            current, fine_time_indices.to(device), stream_id=1
        )
        control_query = self.control_query.reshape(1, -1).to(history.dtype)
        sequence = torch.cat((history, current, control_query), dim=0)
        return sequence.to(self.model.embed_tokens.weight.dtype)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        coarse_tokens = kwargs.get("coarse_tokens")
        if coarse_tokens is None:
            return []
        coarse_time_indices = kwargs["coarse_time_indices"]
        fine_tokens = kwargs["fine_tokens"]
        fine_time_indices = kwargs["fine_time_indices"]
        coarse_lengths = _as_length_list(kwargs["coarse_lengths"])
        fine_lengths = _as_length_list(kwargs["fine_lengths"])

        coarse_tokens = _as_2d(coarse_tokens)
        fine_tokens = _as_2d(fine_tokens)
        coarse_time_indices = _as_1d(coarse_time_indices)
        fine_time_indices = _as_1d(fine_time_indices)

        coarse_tok = torch.split(coarse_tokens, coarse_lengths)
        coarse_time = torch.split(coarse_time_indices, coarse_lengths)
        fine_tok = torch.split(fine_tokens, fine_lengths)
        fine_time = torch.split(fine_time_indices, fine_lengths)

        return [
            self._embed_visual_bundle(ct, cti, ft, fti)
            for ct, cti, ft, fti in zip(coarse_tok, coarse_time, fine_tok, fine_time)
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (name, weight)
            for name, weight in weights
            if not name.startswith("output_scale")
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _as_2d(tensor: object) -> torch.Tensor:
    if isinstance(tensor, (list, tuple)):
        tensor = torch.cat([_as_2d(t) for t in tensor], dim=0)
    assert isinstance(tensor, torch.Tensor)
    return tensor.reshape(-1, tensor.shape[-1])


def _as_1d(tensor: object) -> torch.Tensor:
    if isinstance(tensor, (list, tuple)):
        tensor = torch.cat([_as_1d(t) for t in tensor], dim=0)
    assert isinstance(tensor, torch.Tensor)
    return tensor.reshape(-1)


def _as_length_list(lengths: object) -> list[int]:
    if isinstance(lengths, (list, tuple)):
        flat: list[int] = []
        for item in lengths:
            flat.extend(_as_length_list(item))
        return flat
    assert isinstance(lengths, torch.Tensor)
    return [int(x) for x in lengths.reshape(-1).tolist()]
