
import pytest
import sys
from unittest.mock import MagicMock
import torch

# Mock deep_ep module before importing vllm modules that depend on it
mock_deep_ep = MagicMock()
sys.modules["deep_ep"] = mock_deep_ep

# Mock Buffer class
class MockBuffer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dispatch_count = 0
        self.combine_count = 0

    def low_latency_dispatch(self, *args, **kwargs):
        self.dispatch_count += 1
        # Return dummy values: expert_x, expert_num_tokens, handle, _, hook
        return (
            torch.randn(1, 1, 1), 
            torch.ones(1), 
            f"handle_{id(self)}_{self.dispatch_count}", 
            None, 
            lambda: None
        )

    def low_latency_combine(self, *args, **kwargs):
        self.combine_count += 1
        return None, None, lambda: None

    @staticmethod
    def get_low_latency_rdma_size_hint(*args, **kwargs):
        return 1024

mock_deep_ep.Buffer = MockBuffer

# Now import the class under test
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import DeepEPLLPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

def test_deepep_double_buffering():
    # Setup
    buffer1 = MockBuffer(name="buf1")
    buffer2 = MockBuffer(name="buf2")
    buffers = [buffer1, buffer2]
    
    prepare_finalize = DeepEPLLPrepareAndFinalize(
        buffer=buffers,
        max_tokens_per_rank=1024,
        num_dispatchers=2,
    )
    
    # Mock inputs
    a1 = torch.randn(10, 128)
    topk_ids = torch.randint(0, 8, (10, 1))
    topk_weights = torch.ones(10, 1)
    quant_config = MagicMock(spec=FusedMoEQuantConfig)
    quant_config.quant_dtype = torch.float16
    quant_config.a1_scale = None
    quant_config.a1_gscale = None
    quant_config.a2_scale = None
    
    # Mock ubatch_id context
    with pytest.mock.patch("vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize.dbo_current_ubatch_id", return_value=0):
        # 1. First Prepare -> Should use Buffer 1 (index 0)
        hook1, recv1 = prepare_finalize.prepare_async(
            a1, topk_weights, topk_ids, 8, None, False, quant_config
        )
        assert buffer1.dispatch_count == 1
        assert buffer2.dispatch_count == 0
        
        # 2. Second Prepare -> Should use Buffer 2 (index 1)
        hook2, recv2 = prepare_finalize.prepare_async(
            a1, topk_weights, topk_ids, 8, None, False, quant_config
        )
        assert buffer1.dispatch_count == 1
        assert buffer2.dispatch_count == 1
        
        # 3. First Finalize -> Should use Buffer 1 handle (FIFO)
        # We need mock outputs for finalize
        out = torch.empty_like(a1)
        fused_out = torch.empty_like(a1)
        weight_reduce = MagicMock()
        
        prepare_finalize.finalize_async(
            out, fused_out, topk_weights, topk_ids, False, weight_reduce
        )
        assert buffer1.combine_count == 1
        assert buffer2.combine_count == 0
        
        # 4. Third Prepare -> Should use Buffer 1 (index 0) again
        hook3, recv3 = prepare_finalize.prepare_async(
            a1, topk_weights, topk_ids, 8, None, False, quant_config
        )
        assert buffer1.dispatch_count == 2
        assert buffer2.dispatch_count == 1
        
        # 5. Second Finalize -> Should use Buffer 2 handle
        prepare_finalize.finalize_async(
            out, fused_out, topk_weights, topk_ids, False, weight_reduce
        )
        assert buffer1.combine_count == 1
        assert buffer2.combine_count == 1

if __name__ == "__main__":
    test_deepep_double_buffering()
