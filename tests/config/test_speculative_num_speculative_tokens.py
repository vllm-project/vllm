# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``num_speculative_tokens`` resolution in ``SpeculativeConfig``.

``num_speculative_tokens`` is documented to default to the draft config's
``n_predict`` when omitted. For MTP models this default was unreachable: method
auto-detection compared ``num_speculative_tokens > 1`` before the ``n_predict``
default was applied, so omitting it raised ``TypeError: '>' not supported
between instances of 'NoneType' and 'int'`` (#55323).
"""

from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


def _make_mtp_speculative_config(
    num_speculative_tokens: int | None,
    n_predict: int | None = 1,
) -> SpeculativeConfig:
    """Build an MTP ``SpeculativeConfig`` offline (no draft weights loaded).

    ``ModelConfig`` is patched so the draft config is a synthetic
    ``PretrainedConfig`` whose ``model_type`` is an MTP type; ``n_predict``
    stands in for the value a real draft checkpoint would carry.
    """
    draft_hf_config = PretrainedConfig(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        num_hidden_layers=1,
    )
    if n_predict is not None:
        draft_hf_config.n_predict = n_predict
    draft_model_config = MagicMock(
        model="draft",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    kwargs = dict(
        model="draft",
        method="mtp",
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )
    if num_speculative_tokens is not None:
        kwargs["num_speculative_tokens"] = num_speculative_tokens

    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        return SpeculativeConfig(**kwargs)


@pytest.mark.cpu_test
def test_mtp_num_speculative_tokens_defaults_from_n_predict():
    """Omitting ``num_speculative_tokens`` for an MTP draft defaults it to the
    draft config's ``n_predict`` rather than raising ``TypeError`` (#55323)."""
    spec = _make_mtp_speculative_config(num_speculative_tokens=None, n_predict=2)
    assert spec.num_speculative_tokens == 2


@pytest.mark.cpu_test
def test_mtp_explicit_num_speculative_tokens_is_preserved():
    """An explicit ``num_speculative_tokens`` is honored, not overwritten by
    the ``n_predict`` default."""
    spec = _make_mtp_speculative_config(num_speculative_tokens=1, n_predict=1)
    assert spec.num_speculative_tokens == 1


@pytest.mark.cpu_test
def test_mtp_missing_n_predict_raises_value_error():
    """When neither ``num_speculative_tokens`` nor the draft's ``n_predict`` is
    available, the failure is a clear ``ValueError``, never a ``TypeError``."""
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        _make_mtp_speculative_config(num_speculative_tokens=None, n_predict=None)
