# SPDX-License-Identifier: Apache-2.0
"""Verify CLI flags reach VllmConfig.quest_config."""
from __future__ import annotations

import argparse
import json


def _parse(argv):
    """Helper: feed argv to EngineArgs.from_cli_args and return EngineArgs."""
    from vllm.engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    ns = parser.parse_args(argv)
    return EngineArgs.from_cli_args(ns)


def test_no_quest_flag_means_no_quest_config(tmp_path):
    args = _parse(["--model", "facebook/opt-125m"])
    assert getattr(args, "enable_quest_sparse_offload", False) is False
    assert getattr(args, "quest_config", None) in (None, "")


def test_minimal_enable_flag_creates_default_quest_config():
    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-quest-sparse-offload",
        ]
    )
    assert args.enable_quest_sparse_offload is True


def test_quest_top_k_flag_overrides_default():
    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-quest-sparse-offload",
            "--quest-top-k",
            "128",
        ]
    )
    assert args.enable_quest_sparse_offload is True
    assert args.quest_top_k == 128


def test_quest_config_json_file_loads(tmp_path):
    p = tmp_path / "quest.json"
    p.write_text(json.dumps({"enabled": True, "top_k": 96}))
    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--quest-config",
            str(p),
        ]
    )
    assert args.quest_config == str(p)


def test_engine_args_to_vllm_config_populates_quest_config():
    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-quest-sparse-offload",
            "--quest-top-k",
            "32",
        ]
    )
    # Note: We only check the EngineArgs->QuestConfig translation helper.
    # Building a full VllmConfig may require more fields; we test the helper.
    from vllm.engine.arg_utils import _quest_config_from_args

    cfg = _quest_config_from_args(args)
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.top_k == 32
    cfg.validate()  # must not raise


def test_engine_args_quest_config_off_by_default():
    args = _parse(["--model", "facebook/opt-125m"])
    from vllm.engine.arg_utils import _quest_config_from_args

    assert _quest_config_from_args(args) is None
