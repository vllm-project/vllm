# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shutil
from pathlib import Path

from vllm.plugins.model_metadata import (
    MetadataSource,
    register_model_metadata_provider,
)


class DummyMetadataProvider:
    """Serves non-weight files from a local HF cache and records each call."""

    def __init__(self, store: str, calls: str):
        self.store = Path(store)
        self.calls = Path(calls)

    def prepare_source(self, source: MetadataSource) -> None:
        self._record("source", source)
        repo = "models--" + source.model.replace("/", "--")
        target = Path(source.cache_root) / repo
        if source.offline and (self.store / repo).is_dir() and not target.exists():
            shutil.copytree(self.store / repo, target, symlinks=True)

    def prepare_consumer(
        self, source: MetadataSource, *, model: str, tokenizer: str
    ) -> None:
        self._record("consumer", source, model=model, tokenizer=tokenizer)

    def _record(self, phase: str, source: MetadataSource, **paths: str) -> None:
        call = {"phase": phase, "source_model": source.model, **paths}
        with self.calls.open("a") as f:
            f.write(json.dumps(call) + "\n")


_provider: DummyMetadataProvider | None = None


def register():
    global _provider
    store = os.getenv("VLLM_DUMMY_METADATA_STORE")
    if store is None:
        # Installed alongside other test plugins: act only when a test opts in.
        return
    if _provider is None:
        calls = os.environ["VLLM_DUMMY_METADATA_CALLS"]
        _provider = DummyMetadataProvider(store, calls)
    register_model_metadata_provider(_provider)
