# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import create_new_process_for_each_test
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.common.startup import get_non_default_args
from vllm.plugins.model_metadata import (
    MetadataSource,
    MetadataUnavailable,
    register_model_metadata_provider,
)
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

MODEL = "vllm-project/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def metadata_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A snapshot of `MODEL` without weights, in an HF cache of its own."""
    return Path(
        hf_api().snapshot_download(
            MODEL,
            cache_dir=tmp_path_factory.mktemp("store"),
            ignore_patterns=["*.safetensors"],
        )
    )


class FailingProvider:
    def prepare_source(self, source: MetadataSource) -> None:
        raise OSError("store unreachable")

    def prepare_consumer(
        self, source: MetadataSource, *, model: str, tokenizer: str
    ) -> None:
        raise OSError("store unreachable")


class RecordingProvider:
    def __init__(self) -> None:
        self.sources: list[MetadataSource] = []

    def prepare_source(self, source: MetadataSource) -> None:
        self.sources.append(source)

    def prepare_consumer(
        self, source: MetadataSource, *, model: str, tokenizer: str
    ) -> None:
        pass


def test_offline_engine_gets_metadata_from_provider(
    metadata_snapshot: Path, tmp_path: Path
):
    """Offline with an empty HF cache, the `vllm_add_dummy_metadata_provider`
    plugin supplies the files and is consulted again by the engine process."""
    cache = tmp_path / "hub"
    calls = tmp_path / "calls.jsonl"
    env = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_CACHE": str(cache),
        "VLLM_DUMMY_METADATA_STORE": str(metadata_snapshot.parents[2]),
        "VLLM_DUMMY_METADATA_CALLS": str(calls),
    }
    # Needs a fresh interpreter: offline mode and the cache are read on import.
    code = (
        "from vllm import LLM, EngineArgs, SamplingParams\n"
        f"args = EngineArgs(model={MODEL!r}, load_format='dummy',"
        " gpu_memory_utilization=0.1, enforce_eager=True)\n"
        "llm = LLM.from_engine_args(args)\n"
        "params = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)\n"
        "outputs = llm.generate(['Hello'], params)\n"
        "assert len(outputs[0].outputs[0].token_ids) == 4\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)

    consumers = [
        call
        for call in map(json.loads, calls.read_text().splitlines())
        if call["phase"] == "consumer"
    ]
    # One call from EngineCore and one from its worker.
    assert len(consumers) == 2
    for call in consumers:
        assert call["source_model"] == MODEL
        assert Path(call["model"]).is_relative_to(cache)


@create_new_process_for_each_test()
def test_last_registered_provider_is_used(metadata_snapshot: Path):
    """A second plugin replaces the provider instead of stopping startup."""
    first, last = RecordingProvider(), RecordingProvider()
    register_model_metadata_provider(first)
    register_model_metadata_provider(last)
    EngineArgs(model=str(metadata_snapshot))

    assert not first.sources
    assert [source.model for source in last.sources] == [str(metadata_snapshot)]


@create_new_process_for_each_test()
def test_changed_arguments_drop_source(metadata_snapshot: Path, tmp_path: Path):
    """The consumer phase never sees a source prepared for other arguments."""
    register_model_metadata_provider(RecordingProvider())
    args = EngineArgs(model=str(metadata_snapshot))
    # Creating an engine config resolves the arguments in place; not a change.
    args.create_engine_config()
    source = args.create_model_config().metadata_source
    assert source is not None
    assert source.model == str(metadata_snapshot)

    other = tmp_path / "other"
    shutil.copytree(metadata_snapshot, other)
    args.model = str(other)
    assert args.create_model_config().metadata_source is None


@create_new_process_for_each_test()
def test_argument_logging_skips_source_phase(metadata_snapshot: Path):
    """Reading the default arguments must not ask for the default revision."""
    provider = RecordingProvider()
    register_model_metadata_provider(provider)
    args = EngineArgs(model=str(metadata_snapshot), revision="main")
    assert get_non_default_args(args)["revision"] == "main"

    assert [source.revision for source in provider.sources] == ["main"]


@create_new_process_for_each_test()
def test_consumer_failure_stops_engine_startup(metadata_snapshot: Path):
    """Only the consumer phase is fatal: `EngineArgs` survives the same failure."""
    register_model_metadata_provider(FailingProvider())
    vllm_config = EngineArgs(model=str(metadata_snapshot)).create_engine_config()

    with pytest.raises(MetadataUnavailable, match="store unreachable"):
        EngineCore(vllm_config, Executor.get_class(vllm_config), log_stats=False)
