# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

import vllm.envs as envs
from vllm.models.deepseek_v4 import online_c128
from vllm.models.deepseek_v4.online_c128 import (
    assert_online_c128_supported,
    plan_online_c128_segments,
)


def _config(**kwargs):
    values = dict(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector=None),
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_plan_online_c128_segments_exact_boundary_emits_and_resets():
    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, 128], dtype=np.int32),
        seq_lens_cpu=np.array([128], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert plan.emit_segments.cpu().tolist() == [[0, 128, -1, 127, -1]]
    assert plan.update_segments.shape == (0, 5)
    assert plan.reset_rows.cpu().tolist() == [3]


def test_plan_online_c128_segments_partial_start_splits_emit_and_update():
    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, 3], dtype=np.int32),
        seq_lens_cpu=np.array([130], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert plan.emit_segments.cpu().tolist() == [[0, 1, 3, 0, -1]]
    assert plan.update_segments.cpu().tolist() == [[1, 2, -1, -1, 3]]
    assert plan.reset_rows.cpu().tolist() == []


def test_plan_online_c128_segments_req_mask_skips_unselected_reqs():
    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, 128, 256], dtype=np.int32),
        seq_lens_cpu=np.array([128, 256], dtype=np.int32),
        req_state_indices_cpu=np.array([3, 4], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
        req_mask=np.array([False, True]),
    )

    assert plan.emit_segments.cpu().tolist() == [[128, 128, -1, 255, -1]]
    assert plan.update_segments.shape == (0, 5)
    assert plan.reset_rows.cpu().tolist() == [4]


def test_online_c128_rejects_speculative_decode(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_ONLINE_C128_COMPRESS", True)
    monkeypatch.setattr(online_c128, "_is_sm90", lambda: True)

    with pytest.raises(ValueError, match="does not support speculative decoding"):
        assert_online_c128_supported(
            _config(speculative_config=SimpleNamespace(method="mtp")),
            compress_ratio=128,
            head_dim=512,
        )


def test_online_c128_rejects_wrong_shape(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_ONLINE_C128_COMPRESS", True)
    monkeypatch.setattr(online_c128, "_is_sm90", lambda: True)

    with pytest.raises(ValueError, match="compress_ratio == 128"):
        assert_online_c128_supported(_config(), compress_ratio=4, head_dim=512)

    with pytest.raises(ValueError, match="head_dim == 512"):
        assert_online_c128_supported(_config(), compress_ratio=128, head_dim=128)
