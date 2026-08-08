# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tests for sentence-transformers pooling config parsing,
covering the >= 5.4.0 compact ``1_Pooling/config.json`` schema (#45995)."""

from unittest import mock

import pytest

from vllm.transformers_utils import config as config_mod
from vllm.transformers_utils.config import get_pooling_config, parse_pooling_type


@pytest.fixture(autouse=True)
def _clear_pooling_config_cache():
    # get_pooling_config is @cache'd; clear between parametrized cases.
    get_pooling_config.cache_clear()
    yield
    get_pooling_config.cache_clear()

_MODULES_JSON = [
    {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.models.Transformer",
    },
    {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling",
    },
]

# sentence-transformers <= 5.3.0 ("verbose") schema: one boolean flag per mode.
_VERBOSE_MEAN = {
    "word_embedding_dimension": 384,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_lasttoken": False,
    "include_prompt": True,
}

# sentence-transformers >= 5.4.0 ("compact") schema: single string field.
_COMPACT_MEAN = {
    "embedding_dimension": 384,
    "pooling_mode": "mean",
    "include_prompt": True,
}
_COMPACT_CLS = {"embedding_dimension": 768, "pooling_mode": "cls"}
_COMPACT_LASTTOKEN = {"embedding_dimension": 4096, "pooling_mode": "lasttoken"}


def _patch_pooling_files(pooling_dict):
    """Patch the Hub helpers so get_pooling_config sees the given pooling dict."""

    def fake_get_hf_file_to_dict(file_name, model, revision):
        if file_name == "modules.json":
            return _MODULES_JSON
        if file_name == "1_Pooling/config.json":
            return pooling_dict
        return None

    return (
        mock.patch.object(config_mod, "file_or_path_exists", return_value=True),
        mock.patch.object(
            config_mod, "get_hf_file_to_dict", side_effect=fake_get_hf_file_to_dict
        ),
    )


@pytest.mark.parametrize(
    "pooling_dict, expected",
    [
        (_VERBOSE_MEAN, "MEAN"),
        (_COMPACT_MEAN, "MEAN"),
        (_COMPACT_CLS, "CLS"),
        (_COMPACT_LASTTOKEN, "LAST"),
    ],
)
def test_get_pooling_config_schemas(pooling_dict, expected):
    p1, p2 = _patch_pooling_files(pooling_dict)
    with p1, p2:
        config = get_pooling_config("dummy-model", revision=None)
    assert config is not None
    assert config["seq_pooling_type"] == expected


def _warning_messages(mock_warning):
    return [call.args[0] for call in mock_warning.call_args_list]


def test_compact_schema_warns_when_unparseable():
    # A compact pooling mode that maps to no supported vLLM pooling type
    # (e.g. weighted-mean) must not be applied silently.
    p1, p2 = _patch_pooling_files({"pooling_mode": "weightedmean"})
    # vLLM's logger does not propagate to caplog, so assert on logger.warning.
    with p1, p2, mock.patch.object(config_mod.logger, "warning") as warn:
        config = get_pooling_config("dummy-model", revision=None)
    assert config is not None
    assert "seq_pooling_type" not in config
    assert any(
        "could not derive a supported pooling type" in msg
        for msg in _warning_messages(warn)
    )


def test_valid_config_does_not_warn():
    p1, p2 = _patch_pooling_files(_COMPACT_MEAN)
    with p1, p2, mock.patch.object(config_mod.logger, "warning") as warn:
        get_pooling_config("dummy-model", revision=None)
    assert not any(
        "could not derive a supported pooling type" in msg
        for msg in _warning_messages(warn)
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        # compact (ST >= 5.4.0) strings
        ("mean", "MEAN"),
        ("cls", "CLS"),
        ("max", "MAX"),
        ("lasttoken", "LAST"),
        ("mean_sqrt_len_tokens", "MEAN"),
        # verbose (ST <= 5.3.0) keys
        ("pooling_mode_mean_tokens", "MEAN"),
        ("pooling_mode_cls_token", "CLS"),
        ("pooling_mode_lasttoken", "LAST"),
    ],
)
def test_parse_pooling_type(name, expected):
    assert parse_pooling_type(name) == expected
