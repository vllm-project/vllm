# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

import vllm.plugins as plugins
from vllm.plugins import model_metadata
from vllm.v1.engine.core import EngineCore

if TYPE_CHECKING:
    from vllm.config import VllmConfig

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_metadata_consumer_prepares_local_cache_before_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stop at executor construction, before any device or KV cache setup."""
    cache_root = tmp_path / "cache"
    snapshot = cache_root / "models--org--model" / "snapshots" / ("a" * 40)
    source = model_metadata.MetadataSource(
        model="org/model",
        tokenizer=None,
        revision="release",
        tokenizer_revision=None,
        code_revision=None,
        cache_root=str(cache_root),
        offline=True,
        use_modelscope=False,
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(snapshot),
            tokenizer=str(snapshot),
            metadata_source=source,
        ),
        parallel_config=SimpleNamespace(data_parallel_rank_local=1),
    )
    cache_root.rename(tmp_path / "frontend-cache")
    events: list[str] = []

    def prepare_consumer(received_source, *, model: str, tokenizer: str) -> None:
        assert received_source == source
        assert model == tokenizer == str(snapshot)
        assert not snapshot.exists()
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        events.append("consumer")

    provider = Mock(spec=model_metadata.MetadataProvider)
    provider.prepare_consumer.side_effect = prepare_consumer

    def load_plugins():
        events.append("plugins")
        model_metadata.register_model_metadata_provider(provider)

    monkeypatch.setattr(model_metadata, "_provider", None)
    monkeypatch.setattr(plugins, "load_general_plugins", load_plugins)

    class StopAtExecutor(Exception):
        pass

    def create_executor(vllm_config):
        assert vllm_config is config
        assert (snapshot / "config.json").read_text(encoding="utf-8") == "{}"
        events.append("executor")
        raise StopAtExecutor

    with pytest.raises(StopAtExecutor):
        EngineCore(
            cast("VllmConfig", config),
            executor_class=Mock(side_effect=create_executor),
            log_stats=False,
        )

    assert events == ["plugins", "consumer", "executor"]
