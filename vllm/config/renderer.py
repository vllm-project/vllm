# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Literal

from pydantic import Field, SkipValidation
from pydantic.dataclasses import dataclass

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.transformers_utils.gguf_utils import is_gguf
from vllm.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from vllm.transformers_utils.utils import maybe_model_redirect

TokenizerMode = Literal["auto", "hf", "slow", "mistral", "deepseek_v32"]


@config
@dataclass
class RendererConfig:
    """Configuration for the renderer."""

    # NOTE: In reality, this is a required argument.
    # We provide a dummy default value here to generate the CLI args.
    model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """Provides model context to the renderer."""

    tokenizer: str = ""
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode | str = "auto"
    """Tokenizer mode:\n
    - "auto" will use the tokenizer from `mistral_common` for Mistral models
    if available, otherwise it will use the "hf" tokenizer.\n
    - "hf" will use the fast tokenizer if available.\n
    - "slow" will always use the slow tokenizer.\n
    - "mistral" will always use the tokenizer from `mistral_common`.\n
    - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.\n
    - Other custom values can be supported via plugins."""
    tokenizer_revision: str | None = None
    """The specific revision to use for the tokenizer on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    skip_tokenizer_init: bool = False
    """Skip initialization of tokenizer and detokenizer. Expects valid
    `prompt_token_ids` and `None` for prompt from the input. The generated
    output will contain token ids."""

    io_processor_plugin: str | None = None
    """IOProcessor plugin name to load at model startup."""

    media_io_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Additional args passed to process media inputs, keyed by modalities.
    For example, to set num_frames for video, set
    `--media-io-kwargs '{"video": {"num_frames": 40} }'`"""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    allowed_media_domains: list[str] | None = None
    """If set, only media URLs that belong to this domain can be used for
    multi-modal inputs. """

    @property
    def trust_remote_code(self) -> bool:
        return self.model_config.trust_remote_code

    def __post_init__(self) -> None:
        model_config = self.model_config

        # The tokenizer is consistent with the model by default.
        if not self.tokenizer:
            self.tokenizer = (
                ModelConfig.model
                if model_config is None
                else model_config.original_model
            )
        if not self.tokenizer_revision:
            self.tokenizer_revision = (
                ModelConfig.revision if model_config is None else model_config.revision
            )

        self.original_tokenizer = self.tokenizer
        self.tokenizer = maybe_model_redirect(self.original_tokenizer)
        self.maybe_pull_tokenizer_for_runai(self.tokenizer)

        # Multimodal GGUF models must use original repo for mm processing
        is_multimodal_model = (
            ModelConfig.is_multimodal_model
            if model_config is None
            else model_config.is_multimodal_model
        )
        if is_gguf(self.tokenizer) and is_multimodal_model:
            raise ValueError(
                "Loading a multimodal GGUF model needs to use original "
                "tokenizer. Please specify the unquantized hf model's "
                "repo name or path using the --tokenizer argument."
            )

    def maybe_pull_tokenizer_for_runai(self, tokenizer: str) -> None:
        """Pull tokenizer from Object Storage to temporary directory when needed."""
        if not is_runai_obj_uri(tokenizer):
            return

        object_storage_tokenizer = ObjectStorageModel(url=tokenizer)
        object_storage_tokenizer.pull_files(
            tokenizer,
            ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors", "*.pth"],
        )
        self.tokenizer = object_storage_tokenizer.dir
