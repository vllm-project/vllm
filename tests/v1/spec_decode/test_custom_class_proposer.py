# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import pytest

from vllm.config import VllmConfig
from vllm.v1.spec_decode.custom_class_proposer import create_custom_proposer


class _GoodProposer:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def propose(self, *args, **kwargs):
        return None


class _NoProposeProposer:
    def __init__(self, vllm_config):
        pass


class _NonCallableProposeProposer:
    def __init__(self, vllm_config):
        self.propose = "not callable"


class _RaisingProposer:
    def __init__(self, vllm_config):
        raise RuntimeError("boom")


def _config(model):
    spec = SimpleNamespace(model=model, num_speculative_tokens=2)
    return cast(VllmConfig, SimpleNamespace(speculative_config=spec))


def test_module_path_without_dot_raises_value_error():
    with pytest.raises(ValueError, match="full module path"):
        create_custom_proposer(_config("NoDotPath"))


def test_unimportable_module_raises_import_error():
    with pytest.raises(ImportError, match="Cannot import module"):
        create_custom_proposer(_config("vllm_no_such_module_xyz.MyProposer"))


def test_missing_class_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        create_custom_proposer(_config(f"{__name__}.DoesNotExist"))


def test_constructor_failure_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Failed to instantiate"):
        create_custom_proposer(_config(f"{__name__}._RaisingProposer"))


def test_missing_propose_method_raises_attribute_error():
    with pytest.raises(AttributeError, match="must have a 'propose' method"):
        create_custom_proposer(_config(f"{__name__}._NoProposeProposer"))


def test_non_callable_propose_raises_attribute_error():
    with pytest.raises(AttributeError, match="not callable"):
        create_custom_proposer(_config(f"{__name__}._NonCallableProposeProposer"))


def test_valid_proposer_is_instantiated():
    config = _config(f"{__name__}._GoodProposer")
    proposer = create_custom_proposer(config)
    assert isinstance(proposer, _GoodProposer)
    assert proposer.vllm_config is config
