# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from pydantic import Field
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

    model_config: ModelConfig = Field(default_factory=ModelConfig)
    """Provides model context to the renderer."""

    tokenizer: str = ""
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode | str = "auto"
    """Tokenizer mode:\n
    - "auto" will use "hf" tokenizer if Mistral's tokenizer is not available.\n
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

    # Security-related
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
        # The tokenizer is consistent with the model by default.
        if not self.tokenizer:
            self.tokenizer = self.model_config.model
        if not self.tokenizer_revision:
            self.tokenizer_revision = self.model_config.revision

        self.tokenizer = maybe_model_redirect(self.tokenizer)
        self.maybe_pull_tokenizer_for_runai(self.tokenizer)

        # Multimodal GGUF models must use original repo for mm processing
        if is_gguf(self.tokenizer) and self.model_config.is_multimodal_model:
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
            self.model_config.model,
            ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors", "*.pth"],
        )
        self.tokenizer = object_storage_tokenizer.dir
