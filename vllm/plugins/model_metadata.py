# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class MetadataSource:
    """Inputs passed to an EngineArgs instance for metadata preparation.

    model and tokenizer retain the values passed to that instance. Arguments
    rebuilt by LLM.from_engine_args or default instances for argument logging
    may already contain local paths. Source providers must return without work
    for local paths and repeated calls whose metadata is already prepared.
    """

    model: str
    tokenizer: str | None
    revision: str | None
    tokenizer_revision: str | None
    code_revision: str | None
    cache_root: str
    offline: bool
    use_modelscope: bool


class MetadataUnavailable(RuntimeError):
    """Required model metadata could not be prepared in this process."""


class MetadataProvider(Protocol):
    """Prepare non-weight files without changing native resolution."""

    def prepare_source(self, source: MetadataSource) -> None: ...

    def prepare_consumer(
        self, source: MetadataSource, *, model: str, tokenizer: str
    ) -> None: ...


_provider: MetadataProvider | None = None


def register_model_metadata_provider(provider: MetadataProvider) -> None:
    """Register one provider per process; registering the same object is a no-op."""
    global _provider
    if not all(
        callable(getattr(provider, method, None))
        for method in ("prepare_source", "prepare_consumer")
    ):
        raise TypeError(
            "A model metadata provider must implement callable "
            "prepare_source and prepare_consumer methods."
        )
    if _provider is provider:
        return
    if _provider is not None:
        raise RuntimeError("A model metadata provider is already registered.")
    _provider = provider


def prepare_model_metadata_source(source: MetadataSource) -> MetadataSource | None:
    """Prepare metadata before resolution, leaving failures to native resolution."""
    if _provider is None:
        return None
    try:
        _provider.prepare_source(source)
    except Exception:
        logger.warning(
            "Model metadata preparation failed for %s (revision %s); "
            "continuing with native resolution.",
            source.model,
            source.revision,
            exc_info=True,
        )
    return source


def prepare_model_metadata_consumer(vllm_config: "VllmConfig") -> None:
    """Prepare resolved metadata locally before any engine or worker consumes it."""
    if _provider is None or vllm_config.model_config is None:
        return
    model_config = vllm_config.model_config
    source = model_config.metadata_source
    if source is None:
        return
    try:
        _provider.prepare_consumer(
            source, model=model_config.model, tokenizer=model_config.tokenizer
        )
    except MetadataUnavailable:
        raise
    except Exception as exc:
        raise MetadataUnavailable(
            f"Model metadata preparation failed for {model_config.model!r} "
            f"(revision {source.revision!r}): {exc}"
        ) from exc
