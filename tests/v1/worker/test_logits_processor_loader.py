# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the V2 custom logits processor loader.

Load-time validation is pure Python (importlib + issubclass), so it runs
without CUDA. Pipeline behavior is covered by test_gpu_logits_processors.py.
"""

import subprocess
import sys
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.worker.gpu.sample.logits_processor.loader as loader
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    LogitsProcessor as V1LogitsProcessor,
)
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)


class DummyV2Processor(LogitsProcessor):
    """Records its constructor args so tests can assert how it was built."""

    def __init__(self, vllm_config: Any, req_state: Any):
        self.ctor_args = (vllm_config, req_state)

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        return logits


class AltV2Processor(DummyV2Processor):
    """Second V2 processor, used to assert plugin/FQCN load ordering."""


class DummyV1Processor(V1LogitsProcessor):
    """A V1-interface processor. Never instantiated: the loader must reject
    it at the issubclass check, before any constructor runs."""


def not_a_processor() -> None:
    """FQCN target that resolves but is not a class."""


def _fake_entrypoint(cls: type) -> SimpleNamespace:
    return SimpleNamespace(
        name=cls.__name__.lower(),
        value=f"fake:{cls.__name__}",
        load=lambda: cls,
    )


@pytest.fixture(autouse=True)
def no_installed_plugins(monkeypatch: pytest.MonkeyPatch):
    """Isolate tests from entry points registered in the environment."""
    monkeypatch.setattr(loader, "entry_points", lambda group: [])


class ProcSource(Enum):
    CLASS = auto()
    FQCN = auto()
    ENTRYPOINT = auto()


@pytest.mark.parametrize("source", list(ProcSource))
def test_loads_v2_processors(monkeypatch: pytest.MonkeyPatch, source: ProcSource):
    """A V2 subclass loads via class object, FQCN string, or entrypoint."""
    fake_req_states = SimpleNamespace(
        device="cpu",
        max_num_reqs=4,
        vocab_size=128,
        all_token_ids=None,
        prompt_len=None,
        prefill_len=None,
        total_len=None,
    )
    custom: list[Any]
    if source is ProcSource.CLASS:
        custom = [DummyV2Processor]
    elif source is ProcSource.FQCN:
        custom = [f"{__name__}:DummyV2Processor"]
    else:
        monkeypatch.setattr(
            loader, "entry_points", lambda group: [_fake_entrypoint(DummyV2Processor)]
        )
        # Plugins load before user-specified processors.
        custom = [f"{__name__}:AltV2Processor"]

    procs = loader.build_custom_logits_processors(None, fake_req_states, False, custom)

    if source is ProcSource.ENTRYPOINT:
        assert [type(p) for p in procs] == [DummyV2Processor, AltV2Processor]
    else:
        assert [type(p) for p in procs] == [DummyV2Processor]
    vllm_config, req_state = procs[0].ctor_args
    assert vllm_config is None
    assert isinstance(req_state, loader.LogitsProcRequestState)
    assert req_state.max_num_reqs == 4


@pytest.mark.parametrize(
    ("custom", "plugin_cls", "exc_type", "msg_fragment"),
    [
        pytest.param(
            [DummyV1Processor],
            None,
            ValueError,
            "not a subclass",
            id="v1-class-object",
        ),
        pytest.param(
            [f"{__name__}:DummyV1Processor"],
            None,
            ValueError,
            "not a subclass",
            id="v1-fqcn",
        ),
        pytest.param(
            [],
            DummyV1Processor,
            ValueError,
            "V1-interface plugins are not supported",
            id="v1-entrypoint",
        ),
        pytest.param(
            [f"{__name__}:not_a_processor"],
            None,
            ValueError,
            "must be a type",
            id="non-type-fqcn",
        ),
        pytest.param(
            ["no.such.module:Nope"],
            None,
            RuntimeError,
            "no.such.module:Nope",
            id="missing-module",
        ),
        pytest.param(
            ["no_colon_here"],
            None,
            ValueError,
            "Expected format",
            id="fqcn-missing-colon",
        ),
        pytest.param(
            ["too:many:colons"],
            None,
            ValueError,
            "Expected format",
            id="fqcn-extra-colons",
        ),
    ],
)
def test_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
    custom: list,
    plugin_cls: type | None,
    exc_type: type[Exception],
    msg_fragment: str,
):
    """V1 processors and malformed references fail at load time, not at
    sampling time, with a message naming the offending reference."""
    if plugin_cls is not None:
        monkeypatch.setattr(
            loader, "entry_points", lambda group: [_fake_entrypoint(plugin_cls)]
        )

    with pytest.raises(exc_type, match=msg_fragment):
        loader.build_custom_logits_processors(None, None, False, custom)


def test_pooling_model_rejects_custom_logitsprocs():
    """Pooling models reject custom processors instead of ignoring them."""
    with pytest.raises(ValueError, match="Pooling models do not support"):
        loader.build_custom_logits_processors(None, None, True, [DummyV2Processor])
    assert loader.build_custom_logits_processors(None, None, True, []) == []


class ValidatingProcessor(DummyV2Processor):
    """Rejects target_token=-1 via the optional validate_params hook."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        if (sampling_params.extra_args or {}).get("target_token") == -1:
            raise ValueError("target_token must not be -1")


def test_validate_params_runs_at_admission():
    """The factory loads classes up front; the returned validator runs
    validate_params per request and wraps rejections as VLLMValidationError."""
    loader._cached_load_v2_logitsprocs.cache_clear()
    validate = loader.build_custom_logits_processors_params_validator(
        [ValidatingProcessor]
    )
    validate(SamplingParams(extra_args={"target_token": 1}))
    with pytest.raises(VLLMValidationError, match="target_token"):
        validate(SamplingParams(extra_args={"target_token": -1}))


def test_validator_factory_loads_classes_eagerly():
    """A bad reference fails when the validator is built (at startup), not on
    the first request."""
    loader._cached_load_v2_logitsprocs.cache_clear()
    with pytest.raises(RuntimeError, match="no.such.module:Nope"):
        loader.build_custom_logits_processors_params_validator(["no.such.module:Nope"])


def test_loader_import_stays_frontend_safe():
    """The frontend process imports the loader to validate params; importing
    it must not pull in model-runner side modules."""
    code = (
        "import sys\n"
        "import vllm.v1.worker.gpu.sample.logits_processor.loader\n"
        "assert 'vllm.sampling_params' not in sys.modules\n"
        "assert 'vllm.v1.sample.logits_processor' not in sys.modules\n"
        "assert 'vllm.v1.worker.gpu.states' not in sys.modules\n"
        "assert 'vllm.v1.worker.gpu.buffer_utils' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
