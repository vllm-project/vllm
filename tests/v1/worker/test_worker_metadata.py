# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.plugins as plugins
from vllm.plugins import model_metadata
from vllm.v1.worker import worker_base

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_metadata_consumer_runs_before_worker_import_and_init_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    source = model_metadata.MetadataSource(
        model="org/model",
        tokenizer=None,
        revision="release",
        tokenizer_revision=None,
        code_revision=None,
        cache_root=str(tmp_path),
        offline=True,
        use_modelscope=False,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(snapshot),
            tokenizer=str(snapshot),
            metadata_source=source,
            supports_multimodal_inputs=False,
        ),
        parallel_config=SimpleNamespace(
            worker_cls="tests.fake_worker.FakeWorker",
            worker_extension_cls="",
        ),
        enable_trace_function_call_for_thread=Mock(),
    )
    events: list[str] = []

    def prepare_consumer(received_source, *, model: str, tokenizer: str) -> None:
        assert received_source is source
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

    class FakeWorker:
        def __init__(self, vllm_config):
            assert vllm_config is config
            assert (snapshot / "config.json").read_text(encoding="utf-8") == "{}"
            events.append("worker")

        def init_device(self):
            assert (snapshot / "config.json").read_text(encoding="utf-8") == "{}"
            events.append("device")

    def resolve_worker(qualname: str):
        assert qualname == config.parallel_config.worker_cls
        assert (snapshot / "config.json").read_text(encoding="utf-8") == "{}"
        events.append("import")
        return FakeWorker

    monkeypatch.setattr(model_metadata, "_provider", None)
    monkeypatch.setattr(plugins, "load_general_plugins", load_plugins)
    monkeypatch.setattr(worker_base, "resolve_obj_by_qualname", resolve_worker)

    wrapper = worker_base.WorkerWrapperBase()
    wrapper.init_worker([{"vllm_config": config}])
    wrapper.init_device()

    assert events == ["plugins", "consumer", "import", "worker", "device"]
