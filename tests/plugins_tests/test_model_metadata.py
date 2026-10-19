# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import logging
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from vllm.plugins import model_metadata
from vllm.plugins.model_metadata import (
    MetadataProvider,
    MetadataSource,
    MetadataUnavailable,
    prepare_model_metadata_consumer,
    prepare_model_metadata_source,
    register_model_metadata_provider,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_metadata, "_provider", None)


@pytest.fixture
def source() -> MetadataSource:
    return MetadataSource(
        model="org/model",
        tokenizer="org/tokenizer",
        revision="release",
        tokenizer_revision="tokenizer-release",
        code_revision="code-release",
        cache_root="cache",
        offline=True,
        use_modelscope=False,
    )


@pytest.fixture
def provider() -> Mock:
    return Mock(spec=MetadataProvider)


@pytest.fixture
def consumer_config(source: MetadataSource) -> "VllmConfig":
    return cast(
        "VllmConfig",
        SimpleNamespace(
            model_config=SimpleNamespace(
                metadata_source=source,
                model="resolved/model",
                tokenizer="resolved/tokenizer",
            )
        ),
    )


def test_no_provider_is_a_noop(source: MetadataSource):
    assert prepare_model_metadata_source(source) is None
    prepare_model_metadata_consumer(cast("VllmConfig", object()))


def test_same_provider_registration_is_idempotent(
    provider: Mock, source: MetadataSource
):
    register_model_metadata_provider(provider)
    register_model_metadata_provider(provider)
    provider.prepare_source.assert_not_called()

    assert prepare_model_metadata_source(source) is source
    provider.prepare_source.assert_called_once_with(source)


def test_different_provider_registration_is_rejected(
    provider: Mock, source: MetadataSource
):
    register_model_metadata_provider(provider)
    other_provider = Mock(spec=MetadataProvider)
    with pytest.raises(RuntimeError, match="already registered"):
        register_model_metadata_provider(other_provider)

    prepare_model_metadata_source(source)
    provider.prepare_source.assert_called_once_with(source)
    other_provider.prepare_source.assert_not_called()


@pytest.mark.parametrize("method", ["prepare_source", "prepare_consumer"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "not-callable"])
def test_provider_requires_both_callable_methods(
    provider: Mock, method: str, missing: bool
):
    if missing:
        delattr(provider, method)
    else:
        setattr(provider, method, None)
    with pytest.raises(TypeError, match="callable"):
        register_model_metadata_provider(provider)


def test_metadata_source_deepcopy_preserves_frozen_values(source: MetadataSource):
    restored = copy.deepcopy(source)
    assert restored == source
    assert restored is not source
    assert isinstance(restored, MetadataSource)
    with pytest.raises(FrozenInstanceError):
        restored.__setattr__("model", "other/model")


@pytest.mark.parametrize("error_type", [RuntimeError, MetadataUnavailable])
def test_source_failure_warns_and_preserves_source(
    provider: Mock,
    source: MetadataSource,
    caplog_vllm: pytest.LogCaptureFixture,
    error_type: type[Exception],
):
    error = error_type("metadata service unavailable")
    provider.prepare_source.side_effect = error
    register_model_metadata_provider(provider)

    with caplog_vllm.at_level(logging.WARNING, logger=model_metadata.__name__):
        assert prepare_model_metadata_source(source) is source

    record = caplog_vllm.records[-1]
    assert record.levelno == logging.WARNING
    assert source.model in record.getMessage()
    assert "release" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is error


@pytest.mark.parametrize("model_config", [None, SimpleNamespace(metadata_source=None)])
def test_consumer_without_source_is_a_noop(provider: Mock, model_config):
    register_model_metadata_provider(provider)
    config = cast("VllmConfig", SimpleNamespace(model_config=model_config))
    prepare_model_metadata_consumer(config)
    provider.prepare_consumer.assert_not_called()


def test_consumer_receives_original_source_and_resolved_paths(
    provider: Mock, source: MetadataSource, consumer_config: "VllmConfig"
):
    register_model_metadata_provider(provider)
    prepare_model_metadata_consumer(consumer_config)
    provider.prepare_consumer.assert_called_once_with(
        source, model="resolved/model", tokenizer="resolved/tokenizer"
    )


def test_consumer_metadata_unavailable_is_not_wrapped(
    provider: Mock, consumer_config: "VllmConfig"
):
    error = MetadataUnavailable("metadata is missing")
    provider.prepare_consumer.side_effect = error
    register_model_metadata_provider(provider)

    with pytest.raises(MetadataUnavailable) as exc_info:
        prepare_model_metadata_consumer(consumer_config)
    assert exc_info.value is error


def test_consumer_unexpected_failure_is_wrapped(
    provider: Mock, consumer_config: "VllmConfig"
):
    error = ValueError("invalid metadata")
    provider.prepare_consumer.side_effect = error
    register_model_metadata_provider(provider)

    with pytest.raises(MetadataUnavailable, match="resolved/model") as exc_info:
        prepare_model_metadata_consumer(consumer_config)
    assert exc_info.value.__cause__ is error
    assert "invalid metadata" in str(exc_info.value)
