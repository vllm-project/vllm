# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@dataclass
class BaseDummyOptions:
    """Base options for generating dummy data during profiling."""

    count: int = Field(999, ge=0)


@dataclass(config=ConfigDict(extra="forbid"))
class VideoDummyOptions(BaseDummyOptions):
    """Options for generating dummy video data during profiling."""

    num_frames: int | None = Field(None, gt=0)
    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)


@dataclass(config=ConfigDict(extra="forbid"))
class ImageDummyOptions(BaseDummyOptions):
    """Options for generating dummy image data during profiling."""

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)


@dataclass(config=ConfigDict(extra="forbid"))
class AudioDummyOptions(BaseDummyOptions):
    """Options for generating dummy audio data during profiling."""

    length: int | None = Field(None, gt=0)


MMEncoderTPMode = Literal["weights", "data"]
MMCacheType = Literal["shm", "lru"]
DummyOptions: TypeAlias = (
    BaseDummyOptions | VideoDummyOptions | ImageDummyOptions | AudioDummyOptions
)


@config
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: dict[str, DummyOptions] = Field(default_factory=dict)
    """The maximum number of input items and options allowed per 
        prompt for each modality.
    Defaults to 999 for each modality.

    Legacy format (count only):
        {"image": 16, "video": 2}

    Configurable format (with options):
        {"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}, 
        "image": {"count": 5, "width": 512, "height": 512}}

    Mixed format (combining both):
        {"image": 16, "video": {"count": 1, "num_frames": 32, "width": 512, 
        "height": 512}}
    """
    enable_mm_embeds: bool = False
    """If `True`, enables passing multimodal embeddings:
    for `LLM` class, this refers to tensor inputs under `multi_modal_data`;
    for the OpenAI-compatible server, this refers to chat messages with content
    `"type": "*_embeds"`.

    WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed.
    Only enable this flag for trusted users!"""
    media_io_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Additional args passed to process media inputs, keyed by modalities.
    For example, to set num_frames for video, set
    `--media-io-kwargs '{"video": {"num_frames": 40} }'`"""
    mm_processor_kwargs: dict[str, object] | None = None
    """Arguments to be forwarded to the model's processor for multi-modal data,
    e.g., image processor. Overrides for the multi-modal processor obtained
    from `transformers.AutoProcessor.from_pretrained`.

    The available overrides depend on the model that is being run.

    For example, for Phi-3-Vision:
    `{"num_crops": 4}`."""
    mm_processor_cache_gb: float = Field(default=4, ge=0)
    """The size (in GiB) of the multi-modal processor cache, which is used to
    avoid re-processing past multi-modal inputs.

    This cache is duplicated for each API process and engine core process,
    resulting in a total memory usage of
    `mm_processor_cache_gb * (api_server_count + data_parallel_size)`.

    Set to `0` to disable this cache completely (not recommended)."""
    mm_processor_cache_type: MMCacheType = "lru"
    """Type of cache to use for the multi-modal preprocessor/mapper. If `shm`,
    use shared memory FIFO cache. If `lru`, use mirrored LRU cache."""
    mm_shm_cache_max_object_size_mb: int = Field(default=128, ge=0)
    """Size limit (in MiB) for each object stored in the multi-modal processor
    shared memory cache. Only effective when `mm_processor_cache_type` is
    `"shm"`."""
    mm_encoder_only: bool = False
    """
    When enabled, skips the language component of the model.

    This is usually only valid in disaggregated Encoder process.
    """
    mm_encoder_tp_mode: MMEncoderTPMode = "weights"
    """Indicates how to optimize multi-modal encoder inference using tensor
    parallelism (TP).

    - `"weights"`: Within the same vLLM engine, split the weights of
        each layer across TP ranks. (default TP behavior)\n
    - `"data"`: Within the same vLLM engine, split the batched input data
        across TP ranks to process the data in parallel, while hosting
        the full weights on each TP rank.
        This batch-level DP is not to be confused with API request-level
        DP (which is controlled by `--data-parallel-size`).
        This is only supported on a per-model basis and falls back to
        `"weights"` if the encoder does not support DP."""
    mm_encoder_attn_backend: AttentionBackendEnum | None = None
    """Optional override for the multi-modal encoder attention backend when
    using vision transformers. Accepts any value from
    `vllm.v1.attention.backends.registry.AttentionBackendEnum` (e.g. `FLASH_ATTN`)."""
    interleave_mm_strings: bool = False
    """Enable fully interleaved support for multimodal prompts, while using
    --chat-template-content-format=string."""
    skip_mm_profiling: bool = False
    """When enabled, skips multimodal memory profiling and only profiles with
    language backbone model during engine initialization.

    This reduces engine startup time but shifts the responsibility to users for
    estimating the peak memory usage of the activation of multimodal encoder and
    embedding cache."""
    video_pruning_rate: float | None = Field(default=None, ge=0.0, lt=1.0)
    """Sets pruning rate for video pruning via Efficient Video Sampling.
    Value sits in range [0;1) and determines fraction of media tokens
    from each video to be pruned.
    """

    @field_validator("limit_per_prompt", mode="before")
    @classmethod
    def _validate_limit_per_prompt(
        cls, value: dict[str, int | dict[str, int]]
    ) -> dict[str, DummyOptions]:
        for k, v in value.items():
            # Handle legacy format where only count is specified
            if isinstance(v, int):
                v = {"count": v}
            # Convert to the appropriate DummyOptions subclass
            if k == "video":
                value[k] = VideoDummyOptions(**v)
            elif k == "image":
                value[k] = ImageDummyOptions(**v)
            elif k == "audio":
                value[k] = AudioDummyOptions(**v)
            else:
                value[k] = BaseDummyOptions(**v)
        return value

    @field_validator("mm_encoder_attn_backend", mode="before")
    @classmethod
    def _validate_mm_encoder_attn_backend(
        cls, value: str | AttentionBackendEnum | None
    ) -> AttentionBackendEnum | None:
        if isinstance(value, str) and value.upper() == "XFORMERS":
            raise ValueError(
                "Attention backend 'XFORMERS' has been removed (See PR #29262 for "
                "details). Please select a supported attention backend."
            )

        if value is None or isinstance(value, AttentionBackendEnum):
            return value

        assert isinstance(value, str), (
            "mm_encoder_attn_backend must be a string or an AttentionBackendEnum."
        )
        return AttentionBackendEnum[value.upper()]

    @model_validator(mode="after")
    def _validate_multimodal_config(self):
        if self.mm_processor_cache_type != "shm" and (
            self.mm_shm_cache_max_object_size_mb
            != MultiModalConfig.mm_shm_cache_max_object_size_mb
        ):
            raise ValueError(
                "'mm_shm_cache_max_object_size_mb' should only be set when "
                "'mm_processor_cache_type' is 'shm'."
            )
        return self

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = [
            self.mm_encoder_attn_backend.name
            if self.mm_encoder_attn_backend is not None
            else None,
            self.mm_encoder_tp_mode,
        ]
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def get_limit_per_prompt(self, modality: str) -> int:
        """
        Get the maximum number of input items allowed per prompt
        for the given modality (backward compatible).
        """
        limit_data = self.limit_per_prompt.get(modality)

        if limit_data is None:
            # Unspecified modality is set to 999 by default
            return 999
        return limit_data.count

    def get_dummy_options(self, modality: str) -> BaseDummyOptions | None:
        """
        Get the configurable dummy data options for a modality.
        Returns None if no options are configured for this modality.
        """
        # All values are now DummyOptions after normalization
        return self.limit_per_prompt.get(modality)

    def merge_mm_processor_kwargs(
        self,
        inference_kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        """
        Get the keyword arguments to pass to the multi-modal processor
        according to the extra arguments passed during inference.
        """
        kwargs = self.mm_processor_kwargs or {}
        return kwargs | dict(inference_kwargs)

    def is_multimodal_pruning_enabled(self):
        return self.video_pruning_rate is not None and self.video_pruning_rate > 0
