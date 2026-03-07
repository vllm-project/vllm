
import pytest
import torch
from unittest.mock import MagicMock
import sys
import traceback

# Mock deep_ep module
mock_deep_ep = MagicMock()
sys.modules["deep_ep"] = mock_deep_ep
mock_deep_ep.Buffer = MagicMock()

# Mock vllm.logger to avoid initialization issues
mock_logger = MagicMock()
sys.modules["vllm.logger"] = mock_logger
mock_logger.init_logger.return_value = MagicMock()

# Mock vllm.envs
sys.modules["vllm.envs"] = MagicMock()

# Mock other dependencies
sys.modules["vllm.v1.worker.ubatching"] = MagicMock()
sys.modules["vllm.model_executor.layers.fused_moe.utils"] = MagicMock()
sys.modules["vllm.model_executor.layers.fused_moe.topk_weight_and_reduce"] = MagicMock()
sys.modules["vllm._custom_ops"] = MagicMock()

# Mock modular_kernel entirely
mk_mock = MagicMock()
sys.modules["vllm.model_executor.layers.fused_moe.modular_kernel"] = mk_mock
# Ensure base class is a type so inheritance works
class MockBase:
    def __init__(self): pass
mk_mock.FusedMoEPrepareAndFinalizeModular = MockBase
mk_mock.FusedMoEActivationFormat = MagicMock()

try:
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import DeepEPLLPrepareAndFinalize
except Exception:
    traceback.print_exc()
    sys.exit(1)

def test_deepep_identity_optimization():
    print("Starting test...")
    # Setup
    num_experts = 8
    buffer = mock_deep_ep.Buffer()
    max_tokens = 1024
    num_dispatchers = 1
    
    # Case 1: Identity Mapping
    global_to_physical = torch.arange(num_experts, dtype=torch.int32, device="cpu")
    physical_to_global = torch.arange(num_experts, dtype=torch.int32, device="cpu")
    local_expert_global_ids = torch.arange(num_experts, dtype=torch.int32, device="cpu")
    
    # Move to CUDA if available, else simulate with CPU (the logic is device agnostic but requires same device)
    # We test on CPU for simplicity as the logic is just tensor comparison
    
    pf = DeepEPLLPrepareAndFinalize(
        buffer=buffer,
        max_tokens_per_rank=max_tokens,
        num_dispatchers=num_dispatchers,
        global_to_physical=global_to_physical,
        physical_to_global=physical_to_global,
        local_expert_global_ids=local_expert_global_ids
    )
    
    # Assertions: Should be None
    assert pf.global_to_physical is None
    assert pf.physical_to_global is None
    assert pf.local_expert_global_ids is None
    
    # Verify mapping functions return input directly
    input_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    assert pf._map_global_to_physical_ids(input_ids) is input_ids
    assert pf._map_local_to_global_ids(input_ids) is input_ids

    # Case 2: Non-Identity Mapping
    global_to_physical_shuffled = torch.randperm(num_experts, dtype=torch.int32)
    # Ensure it's not identity
    while torch.equal(global_to_physical_shuffled, torch.arange(num_experts, dtype=torch.int32)):
         global_to_physical_shuffled = torch.randperm(num_experts, dtype=torch.int32)
         
    pf_shuffled = DeepEPLLPrepareAndFinalize(
        buffer=buffer,
        max_tokens_per_rank=max_tokens,
        num_dispatchers=num_dispatchers,
        global_to_physical=global_to_physical_shuffled,
    )
    
    # Assertions: Should NOT be None
    assert pf_shuffled.global_to_physical is not None
    assert torch.equal(pf_shuffled.global_to_physical, global_to_physical_shuffled)
    
    # Verify mapping function uses the map
    mapped = pf_shuffled._map_global_to_physical_ids(input_ids)
    expected = global_to_physical_shuffled[input_ids]
    assert torch.equal(mapped, expected)

if __name__ == "__main__":
    test_deepep_identity_optimization()
