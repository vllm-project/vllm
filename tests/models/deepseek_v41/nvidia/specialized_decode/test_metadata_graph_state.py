# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which deployments record the attention metadata builds of FULL decode steps
on SM80 and SM90, and that every metadata builder of their layers is one the
recording accepts."""

from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v41.compressor import CompressorBackend
from vllm.models.deepseek_v41.nvidia import specialized_decode
from vllm.models.deepseek_v41.nvidia.specialized_decode import ampere, hopper
from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.attention import (
    DeepseekV41AmpereAttention,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.indexer_metadata import (  # noqa: E501
    AmpereIndexerBackend,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.attention import (
    DeepseekV41HopperAttention,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.metadata_graphs import (  # noqa: E501
    _RECORDABLE_BUILDERS,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend


def _config(mode=CUDAGraphMode.FULL_AND_PIECEWISE, ubatching=False, dcp=1, pcp=1):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        parallel_config=SimpleNamespace(
            use_ubatching=ubatching,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


@pytest.mark.parametrize("arch", ["sm80", "sm90", "other"])
def test_sm80_and_sm90_record(monkeypatch, arch):
    monkeypatch.setattr(ampere, "is_ampere", lambda: arch == "sm80")
    monkeypatch.setattr(hopper, "is_hopper", lambda: arch == "sm90")
    assert specialized_decode.records_decode_metadata(_config()) == (arch != "other")


@pytest.mark.parametrize(
    "config",
    [
        _config(mode=CUDAGraphMode.PIECEWISE),
        _config(mode=CUDAGraphMode.NONE),
        _config(ubatching=True),
        _config(dcp=2),
        _config(pcp=2),
    ],
)
def test_other_deployments_build_eagerly(monkeypatch, config):
    monkeypatch.setattr(hopper, "is_hopper", lambda: True)
    assert not specialized_decode.records_decode_metadata(config)


@pytest.mark.parametrize(
    "backend",
    [
        DeepseekV41AmpereAttention.backend_cls,
        DeepseekV41AmpereAttention.swa_backend_cls,
        AmpereIndexerBackend,
        DeepseekV41HopperAttention.backend_cls,
        DeepseekV41HopperAttention.swa_backend_cls,
        # The SM90 indexer keeps the stock backend without sparse logits.
        DeepseekV41IndexerBackend,
        CompressorBackend,
    ],
)
def test_sm80_and_sm90_builders_are_recordable(backend):
    assert backend.get_builder_cls() in _RECORDABLE_BUILDERS
