# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from vllm.config.utils import CompileFactors, config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.model_loader import LoadFormats
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
else:
    LoadFormats = Any
    TensorizerConfig = Any

logger = init_logger(__name__)


@config
class LoadConfig:
    """Configuration for loading the model weights."""

    load_format: str | LoadFormats = "auto"
    """The format of the model weights to load:\n
    - "auto" will try to load the weights in the safetensors format and fall
    back to the pytorch bin format if safetensors format is not available.\n
    - "pt" will load the weights in the pytorch bin format.\n
    - "safetensors" will load the weights in the safetensors format.\n
    - "npcache" will load the weights in pytorch format and store a numpy cache
    to speed up the loading.\n
    - "dummy" will initialize the weights with random values, which is mainly
    for profiling.\n
    - "tensorizer" will use CoreWeave's tensorizer library for fast weight
    loading. See the Tensorize vLLM Model script in the Examples section for
    more information.\n
    - "runai_streamer" will load the Safetensors weights using Run:ai Model
    Streamer.\n
    - "runai_streamer_sharded" will load weights from pre-sharded checkpoint
    files using Run:ai Model Streamer.\n
    - "bitsandbytes" will load the weights using bitsandbytes quantization.\n
    - "sharded_state" will load weights from pre-sharded checkpoint files,
    supporting efficient loading of tensor-parallel models.\n
    - "gguf" will load weights from GGUF format files (details specified in
    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).\n
    - "mistral" will load weights from consolidated safetensors files used by
    Mistral models.
    - Other custom values can be supported via plugins."""
    download_dir: str | None = None
    """Directory to download and load the weights, default to the default
    cache directory of Hugging Face."""
    safetensors_load_strategy: str = "lazy"
    """Specifies the loading strategy for safetensors weights.
    - "lazy" (default): Weights are memory-mapped from the file. This enables
      on-demand loading and is highly efficient for models on local storage.
    - "eager": The entire file is read into CPU memory upfront before loading.
      This is recommended for models on network filesystems (e.g., Lustre, NFS)
      as it avoids inefficient random reads, significantly speeding up model
      initialization. However, it uses more CPU RAM.
    - "torchao": Weights are loaded in upfront and then reconstructed
      into torchao tensor subclasses. This is used when the checkpoint
      was quantized using torchao and saved using safetensors.
      Needs torchao >= 0.14.0
    """
    model_loader_extra_config: dict | TensorizerConfig = Field(default_factory=dict)
    """Extra config for model loader. This will be passed to the model loader
    corresponding to the chosen load_format."""
    device: str | None = None
    """Device to which model weights will be loaded, default to
    device_config.device"""
    ignore_patterns: list[str] | str = Field(default_factory=lambda: ["original/**/*"])
    """The list of patterns to ignore when loading the model. Default to
    "original/**/*" to avoid repeated loading of llama's checkpoints."""
    use_tqdm_on_load: bool = True
    """Whether to enable tqdm for showing progress bar when loading model
    weights."""
    pt_load_map_location: str | dict[str, str] = "cpu"
    """
    pt_load_map_location: the map location for loading pytorch checkpoint, to
    support loading checkpoints can only be loaded on certain devices like
    "cuda", this is equivalent to {"": "cuda"}. Another supported format is
    mapping from different devices like from GPU 1 to GPU 0:
    {"cuda:1": "cuda:0"}. Note that when passed from command line, the strings
    in dictionary needs to be double quoted for json parsing. For more details,
    see original doc for `map_location` in https://pytorch.org/docs/stable/generated/torch.load.html
    """

    def compile_factors(self) -> CompileFactors:
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
        # This config does not affect the compiled graph.
        return {}

    @field_validator("load_format", mode="after")
    def _lowercase_load_format(cls, load_format: str) -> str:
        return load_format.lower()

    @field_validator("ignore_patterns", mode="after")
    def _validate_ignore_patterns(
        cls, ignore_patterns: list[str] | str
    ) -> list[str] | str:
        if ignore_patterns != ["original/**/*"] and len(ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                ignore_patterns,
            )

        return ignore_patterns
