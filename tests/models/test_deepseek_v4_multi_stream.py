# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from vllm.models.deepseek_v4.multi_stream import (
    create_dsv4_aux_stream_list,
    should_overlap_dsv4_indexer,
    should_overlap_dsv4_input_gemms,
)
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


@dataclass
class _FakeSWAMetadata:
    num_decodes: int


def test_create_dsv4_aux_stream_list_cuda():
    with patch("vllm.models.deepseek_v4.multi_stream.current_platform") as platform:
        platform.is_rocm.return_value = False
        platform.is_cuda.return_value = True
        streams = create_dsv4_aux_stream_list()
    assert streams is not None
    assert len(streams) == 3


def test_create_dsv4_aux_stream_list_rocm_disabled():
    with patch("vllm.models.deepseek_v4.multi_stream.current_platform") as platform:
        platform.is_rocm.return_value = True
        with patch("vllm.models.deepseek_v4.multi_stream.envs") as envs:
            envs.VLLM_DSV4_ROCM_MULTI_STREAM = False
            assert create_dsv4_aux_stream_list() is None


def test_create_dsv4_aux_stream_list_rocm_enabled():
    with patch("vllm.models.deepseek_v4.multi_stream.current_platform") as platform:
        platform.is_rocm.return_value = True
        with patch("vllm.models.deepseek_v4.multi_stream.envs") as envs:
            envs.VLLM_DSV4_ROCM_MULTI_STREAM = True
            streams = create_dsv4_aux_stream_list()
    assert streams is not None
    assert len(streams) == 3


def test_should_overlap_dsv4_indexer_rocm_decode_only():
    attn_metadata = {"swa.prefix": _FakeSWAMetadata(num_decodes=2)}
    aux_streams = [torch.cuda.Stream()]
    with patch("vllm.models.deepseek_v4.multi_stream.current_platform") as platform:
        platform.is_rocm.return_value = True
        with patch("vllm.models.deepseek_v4.multi_stream.envs") as envs:
            envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY = True
            assert should_overlap_dsv4_indexer(
                aux_streams, attn_metadata, "swa.prefix"
            )
            assert not should_overlap_dsv4_indexer(
                aux_streams, attn_metadata, "missing.prefix"
            )
            assert not should_overlap_dsv4_indexer(
                aux_streams,
                {"swa.prefix": _FakeSWAMetadata(num_decodes=0)},
                "swa.prefix",
            )


def test_should_overlap_dsv4_input_gemms_respects_threshold():
    aux_streams = [torch.cuda.Stream()]
    attn_metadata = {"swa.prefix": _FakeSWAMetadata(num_decodes=1)}
    with patch("vllm.models.deepseek_v4.multi_stream.current_platform") as platform:
        platform.is_rocm.return_value = False
        with patch("vllm.models.deepseek_v4.multi_stream.envs") as envs:
            envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD = 4
            envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY = True
            assert should_overlap_dsv4_input_gemms(
                4, aux_streams, attn_metadata, "swa.prefix"
            )
            assert not should_overlap_dsv4_input_gemms(
                8, aux_streams, attn_metadata, "swa.prefix"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_maybe_execute_in_parallel_matches_sequential():
    device = torch.device("cuda")
    default_stream = torch.cuda.current_stream(device)
    aux_stream = torch.cuda.Stream(device)
    event0 = torch.cuda.Event()
    event1 = torch.cuda.Event()

    x = torch.randn(32, device=device)
    y = torch.randn(32, device=device)

    def fn0() -> torch.Tensor:
        return x * 2

    def fn1() -> torch.Tensor:
        return y + 1

    parallel = maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream)
    sequential = maybe_execute_in_parallel(fn0, fn1, event0, event1, None)
    torch.cuda.synchronize()

    assert torch.equal(parallel[0], sequential[0])
    assert torch.equal(parallel[1], sequential[1])

    # Aux work should have been queued on the aux stream.
    assert aux_stream != default_stream
