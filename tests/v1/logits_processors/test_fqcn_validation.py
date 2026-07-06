# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns

@pytest.fixture
def should_do_global_cleanup_after_test():
    # Override conftest fixture to avoid PyTorch MPS allocator issues on macOS during teardown
    return False

def test_load_logitsprocs_by_fqcns_invalid_format():
    # Test FQCN missing colon
    with pytest.raises(ValueError) as exc_info:
        _load_logitsprocs_by_fqcns(["math.gcd"])
    assert "Invalid logits processor FQCN" in str(exc_info.value)
    assert "Expected format: <module>:<type>" in str(exc_info.value)

    # Test FQCN with multiple colons
    with pytest.raises(ValueError) as exc_info:
        _load_logitsprocs_by_fqcns(["math:gcd:extra"])
    assert "Invalid logits processor FQCN" in str(exc_info.value)
    assert "Expected format: <module>:<type>" in str(exc_info.value)

def test_load_logitsprocs_by_fqcns_invalid_subclass():
    # Test valid FQCN format but loads a function instead of a LogitsProcessor type.
    # It should pass FQCN parsing, import the module, and then raise ValueError on type check.
    with pytest.raises(ValueError) as exc_info:
        _load_logitsprocs_by_fqcns(["math:gcd"])
    assert "Loaded logit processor must be a type" in str(exc_info.value)
