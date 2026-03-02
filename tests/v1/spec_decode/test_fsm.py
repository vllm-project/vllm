# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile

import numpy as np
import pytest

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.custom_fsm import CustomFSM
from vllm.v1.spec_decode.fsm_proposer import FSMProposer


@pytest.fixture
def simple_fsm():
    """Create a simple FSM for testing."""
    fsm = CustomFSM()
    # State 0 -> token 1 -> State 1
    # State 1 -> token 2 -> State 2
    # State 2 -> token 3 -> State 3
    fsm.graph = {
        0: {1: 1},
        1: {2: 2},
        2: {3: 3},
    }
    return fsm


@pytest.fixture
def fsm_with_wildcard():
    """Create FSM with wildcard transitions."""
    fsm = CustomFSM()
    # State 0 -> token 1 -> State 1
    # State 1 -> wildcard (-1) -> State 2 (freeform)
    # State 2 -> token 5 -> State 3
    fsm.graph = {
        0: {1: 1},
        1: {-1: 2},
        2: {5: 3},
    }
    return fsm


def test_fsm_get_next_tokens(simple_fsm):
    """Test getting valid next tokens from FSM."""
    assert simple_fsm.get_next_tokens([]) == [1]
    assert simple_fsm.get_next_tokens([1]) == [2]
    assert simple_fsm.get_next_tokens([1, 2]) == [3]
    assert simple_fsm.get_next_tokens([1, 2, 3]) == []


def test_fsm_get_next_tokens_wildcard(fsm_with_wildcard):
    """Test wildcard transitions return empty list (all tokens allowed)."""
    assert fsm_with_wildcard.get_next_tokens([1]) == []
    assert fsm_with_wildcard.get_next_tokens([1, 99]) == [5]


def test_fsm_get_next_state(simple_fsm):
    """Test state transitions."""
    assert simple_fsm.get_next_state(0, 1) == 1
    assert simple_fsm.get_next_state(1, 2) == 2
    assert simple_fsm.get_next_state(2, 3) == 3
    assert simple_fsm.get_next_state(3, 1) is None


def test_fsm_save_load():
    """Test FSM serialization."""
    fsm = CustomFSM()
    fsm.graph = {0: {1: 1, 2: 2}, 1: {3: 2}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        fsm.save(f.name)
        f.flush()
        loaded_fsm = CustomFSM.from_prebuilt(f.name)

    assert loaded_fsm.graph == fsm.graph


def test_fsm_proposer_basic():
    """Test FSM proposer with deterministic path."""
    fsm = CustomFSM()
    fsm.graph = {
        0: {1: 1},
        1: {2: 2},
        2: {3: 3},
        3: {4: 4},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        fsm.save(f.name)
        f.flush()

        model_config = ModelConfig(model="facebook/opt-125m")
        proposer = FSMProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    fsm_path=f.name,
                    num_speculative_tokens=3,
                    method="fsm",
                ),
            ),
            fsm=fsm,
        )

    # Single request, deterministic path
    result = proposer.propose(
        sampled_token_ids=[[1]],
        req_ids=["req_0"],
        num_tokens_no_spec=np.array([1]),
        token_ids_cpu=np.array([[1]]),
    )
    assert result == [[2, 3, 4]]


def test_fsm_proposer_no_drafts():
    """Test FSM proposer when no deterministic path exists."""
    fsm = CustomFSM()
    fsm.graph = {
        0: {1: 1, 2: 1},  # Multiple choices, not deterministic
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        fsm.save(f.name)
        f.flush()

        model_config = ModelConfig(model="facebook/opt-125m")
        proposer = FSMProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    fsm_path=f.name,
                    num_speculative_tokens=2,
                    method="fsm",
                ),
            ),
            fsm=fsm,
        )

    result = proposer.propose(
        sampled_token_ids=[[1]],
        req_ids=["req_0"],
        num_tokens_no_spec=np.array([1]),
        token_ids_cpu=np.array([[1]]),
    )
    assert result == [[]]


def test_fsm_proposer_multibatch():
    """Test FSM proposer with multiple requests."""
    fsm = CustomFSM()
    fsm.graph = {
        0: {1: 1},
        1: {2: 2},
        2: {3: 3},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        fsm.save(f.name)
        f.flush()

        model_config = ModelConfig(model="facebook/opt-125m")
        proposer = FSMProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    fsm_path=f.name,
                    num_speculative_tokens=2,
                    method="fsm",
                ),
            ),
            fsm=fsm,
        )

    result = proposer.propose(
        sampled_token_ids=[[1], [2]],
        req_ids=["req_0", "req_1"],
        num_tokens_no_spec=np.array([1, 1]),
        token_ids_cpu=np.array([[1], [2]]),
    )
    assert result == [[2, 3], []]


def test_fsm_proposer_state_tracking():
    """Test FSM proposer maintains state across calls."""
    fsm = CustomFSM()
    fsm.graph = {
        0: {1: 1},
        1: {2: 2},
        2: {3: 3},
        3: {4: 4},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        fsm.save(f.name)
        f.flush()

        model_config = ModelConfig(model="facebook/opt-125m")
        proposer = FSMProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    fsm_path=f.name,
                    num_speculative_tokens=2,
                    method="fsm",
                ),
            ),
            fsm=fsm,
        )

    # First call
    result = proposer.propose(
        sampled_token_ids=[[1]],
        req_ids=["req_0"],
        num_tokens_no_spec=np.array([1]),
        token_ids_cpu=np.array([[1]]),
    )
    assert result == [[2, 3]]

    # Second call - state should be maintained
    result = proposer.propose(
        sampled_token_ids=[[2]],
        req_ids=["req_0"],
        num_tokens_no_spec=np.array([2]),
        token_ids_cpu=np.array([[1, 2]]),
    )
    assert result == [[3, 4]]


def test_fsm_proposer_cleanup():
    """Test FSM proposer cleans up finished requests."""
    fsm = CustomFSM()
    fsm.graph = {0: {1: 1}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        fsm.save(f.name)
        f.flush()

        model_config = ModelConfig(model="facebook/opt-125m")
        proposer = FSMProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    fsm_path=f.name,
                    num_speculative_tokens=2,
                    method="fsm",
                ),
            ),
            fsm=fsm,
        )

    proposer.propose(
        sampled_token_ids=[[1]],
        req_ids=["req_0"],
        num_tokens_no_spec=np.array([1]),
        token_ids_cpu=np.array([[1]]),
    )
    assert "req_0" in proposer.req_states

    proposer.cleanup_finished_requests({"req_0"})
    assert "req_0" not in proposer.req_states


def test_speculative_config_fsm_validation():
    """Test SpeculativeConfig validates fsm_path."""
    # Should raise error when fsm_path is None
    with pytest.raises(ValueError, match="fsm_path must be provided"):
        SpeculativeConfig(
            method="fsm",
            num_speculative_tokens=2,
        )

    # Should succeed with fsm_path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump({}, f)
        f.flush()
        config = SpeculativeConfig(
            method="fsm",
            num_speculative_tokens=2,
            fsm_path=f.name,
        )
        assert config.fsm_path == f.name
