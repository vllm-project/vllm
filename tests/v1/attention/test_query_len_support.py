# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `QueryLenSupport` -> `reorder_batch_threshold` mapping and the
`AttentionRole` foundations (PR1, Part A). Pure CPU, no GPU required."""

import types

import pytest

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    AttentionRole,
    QueryLenSupport,
)


class _DummyBuilder(AttentionMetadataBuilder):
    """Minimal concrete builder that skips the heavy base ``__init__`` and only
    carries what ``_init_reorder_from_query_len_support`` reads."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def build(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


def _make_vllm_config(num_spec=None, parallel_drafting=False, dcp_size=1):
    spec = None
    if num_spec is not None:
        spec = types.SimpleNamespace(
            num_speculative_tokens=num_spec,
            parallel_drafting=parallel_drafting,
        )
    return types.SimpleNamespace(
        speculative_config=spec,
        parallel_config=types.SimpleNamespace(
            decode_context_parallel_size=dcp_size,
        ),
    )


def _resolve_threshold(
    query_len_support,
    reorder_batch_threshold=1,
    num_spec=None,
    parallel_drafting=False,
    dcp_size=1,
    supports_dcp_with_varlen=False,
):
    builder = _DummyBuilder(_make_vllm_config(num_spec, parallel_drafting, dcp_size))
    builder.query_len_support = query_len_support
    builder._init_reorder_from_query_len_support(
        reorder_batch_threshold, supports_dcp_with_varlen
    )
    return builder.reorder_batch_threshold


@pytest.mark.parametrize("num_spec", [None, 1, 3])
def test_single_only_always_threshold_one(num_spec):
    assert _resolve_threshold(QueryLenSupport.SINGLE_ONLY, num_spec=num_spec) == 1


def test_single_only_rejects_threshold_above_one():
    with pytest.raises(AssertionError):
        _resolve_threshold(QueryLenSupport.SINGLE_ONLY, reorder_batch_threshold=2)


@pytest.mark.parametrize(
    "query_len_support", [QueryLenSupport.UNIFORM, QueryLenSupport.VARLEN]
)
@pytest.mark.parametrize(
    "num_spec, expected",
    [
        (None, 1),  # no spec config -> base threshold unchanged
        (1, 2),  # 1 + 1 * 1
        (3, 4),  # 1 + 1 * 3
    ],
)
def test_spec_bump_uniform_and_varlen(query_len_support, num_spec, expected):
    assert _resolve_threshold(query_len_support, num_spec=num_spec) == expected


@pytest.mark.parametrize(
    "num_spec, expected",
    [
        (1, 3),  # 1 + 2 * 1
        (3, 7),  # 1 + 2 * 3
    ],
)
def test_spec_bump_parallel_drafting(num_spec, expected):
    assert (
        _resolve_threshold(
            QueryLenSupport.UNIFORM, num_spec=num_spec, parallel_drafting=True
        )
        == expected
    )


def test_dcp_without_varlen_support_forces_threshold_one():
    # DCP > 1 forces threshold 1 unless the builder supports varlen DCP.
    assert (
        _resolve_threshold(
            QueryLenSupport.UNIFORM,
            num_spec=3,
            dcp_size=2,
            supports_dcp_with_varlen=False,
        )
        == 1
    )


def test_dcp_with_varlen_support_keeps_bumped_threshold():
    assert (
        _resolve_threshold(
            QueryLenSupport.VARLEN,
            num_spec=3,
            dcp_size=2,
            supports_dcp_with_varlen=True,
        )
        == 4
    )


def test_base_builder_default_is_single_only():
    assert AttentionMetadataBuilder.query_len_support == QueryLenSupport.SINGLE_ONLY


def test_query_len_support_reexport_identity():
    from vllm.model_executor.layers.attention.mla_attention import (
        QueryLenSupport as MLAQueryLenSupport,
    )

    assert MLAQueryLenSupport is QueryLenSupport


def test_attention_role_members():
    assert {r.value for r in AttentionRole} == {"prefill", "decode"}


def test_supports_role_defaults_true():
    assert AttentionBackend.supports_role(AttentionRole.PREFILL)
    assert AttentionBackend.supports_role(AttentionRole.DECODE)
    assert AttentionBackend.default_priority(AttentionRole.DECODE) is None
