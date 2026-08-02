# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers import litedsa


@pytest.fixture(autouse=True)
def clear_litedsa_output_buffer_cache():
    with litedsa._OUTPUT_BUFS_LOCK:
        litedsa._OUTPUT_BUFS.clear()
    yield
    with litedsa._OUTPUT_BUFS_LOCK:
        litedsa._OUTPUT_BUFS.clear()


def test_geometric_capacity():
    assert litedsa._geometric_capacity(1) == 1
    assert litedsa._geometric_capacity(3) == 4
    assert litedsa._geometric_capacity(512) == 512
    with pytest.raises(ValueError):
        litedsa._geometric_capacity(0)


def test_union_seq_space_rounding_and_fixed_ab_mode(monkeypatch):
    monkeypatch.setattr(litedsa, "_DYNAMIC_SEQ_SPACE", True)
    assert litedsa._union_seq_space(1) == 1024
    assert litedsa._union_seq_space(196609) == 197632
    assert litedsa._union_seq_space(1 << 20) == 1 << 20
    with pytest.raises(ValueError):
        litedsa._union_seq_space(0)
    with pytest.raises(ValueError):
        litedsa._union_seq_space((1 << 20) + 1)

    monkeypatch.setattr(litedsa, "_DYNAMIC_SEQ_SPACE", False)
    assert litedsa._union_seq_space(196609) == 1 << 20


def test_output_buffers_reuse_and_grow_on_cpu():
    device = torch.device("cpu")
    out1, max1, lse1 = litedsa._get_or_create_output_buffers(
        device, stream_handle=11, owner_thread=22, ng=3
    )
    assert out1.shape == (3, 128, 512)
    assert max1.shape == lse1.shape == (3, 128)
    assert out1.is_contiguous()
    assert max1.is_contiguous()
    assert lse1.is_contiguous()

    out2, max2, lse2 = litedsa._get_or_create_output_buffers(
        device, stream_handle=11, owner_thread=22, ng=2
    )
    assert out2.data_ptr() == out1.data_ptr()
    assert max2.data_ptr() == max1.data_ptr()
    assert lse2.data_ptr() == lse1.data_ptr()

    out3, max3, lse3 = litedsa._get_or_create_output_buffers(
        device, stream_handle=11, owner_thread=22, ng=5
    )
    assert out3.shape == (5, 128, 512)
    assert max3.shape == lse3.shape == (5, 128)
    assert out3.data_ptr() != out1.data_ptr()
    key = litedsa._output_buffer_key(device, 11, 22)
    assert litedsa._OUTPUT_BUFS[key][0] == 8


def test_output_buffers_do_not_alias_streams_or_host_threads():
    device = torch.device("cpu")
    out_a, _, _ = litedsa._get_or_create_output_buffers(device, 1, 7, 1)
    out_b, _, _ = litedsa._get_or_create_output_buffers(device, 2, 7, 1)
    out_c, _, _ = litedsa._get_or_create_output_buffers(device, 1, 8, 1)
    assert len({out_a.data_ptr(), out_b.data_ptr(), out_c.data_ptr()}) == 3


def test_output_buffer_pool_count_is_lru_bounded(monkeypatch):
    monkeypatch.setattr(litedsa, "_OUTPUT_BUF_CACHE_MAX_POOLS", 2)
    device = torch.device("cpu")
    litedsa._get_or_create_output_buffers(device, 1, 7, 1)
    litedsa._get_or_create_output_buffers(device, 2, 7, 1)
    # Touch stream 1, then insert stream 3: stream 2 must be evicted.
    litedsa._get_or_create_output_buffers(device, 1, 7, 1)
    litedsa._get_or_create_output_buffers(device, 3, 7, 1)

    assert len(litedsa._OUTPUT_BUFS) == 2
    assert litedsa._output_buffer_key(device, 1, 7) in litedsa._OUTPUT_BUFS
    assert litedsa._output_buffer_key(device, 2, 7) not in litedsa._OUTPUT_BUFS
    assert litedsa._output_buffer_key(device, 3, 7) in litedsa._OUTPUT_BUFS


def test_reuse_is_disabled_for_autograd_and_capture(monkeypatch):
    monkeypatch.setattr(litedsa, "_REUSE_OUTPUT_BUFS", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.enable_grad():
        assert not litedsa._reuse_output_buffers_allowed()
    with torch.no_grad():
        assert litedsa._reuse_output_buffers_allowed()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.no_grad(), pytest.raises(RuntimeError, match="capture-safe"):
        litedsa._reuse_output_buffers_allowed()
    # The public helper checks before dereferencing inputs or launching union.
    with torch.no_grad(), pytest.raises(RuntimeError, match="capture-safe"):
        litedsa.litedsa_masked_mqa(None, None, None, None, None, 0, 0, 0.0, 0.0, 0, 0)


def test_disabled_env_flag_does_not_query_cuda_capture(monkeypatch):
    monkeypatch.setattr(litedsa, "_REUSE_OUTPUT_BUFS", False)

    def unexpected_capture_query():
        raise AssertionError("capture state must not be queried when reuse is off")

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", unexpected_capture_query
    )
    assert not litedsa._reuse_output_buffers_allowed()
