import pytest
import torch
from vllm import LLM, SamplingParams

def test_layer_skip_smoke_opt(tmp_path):
    """Test with OPT-125M."""
    # Create dummy LSQ head with correct dtype
    head_dir = tmp_path / "heads"
    head_dir.mkdir()
    # Use small dummy head for CI - shape doesn't matter for smoke test
    # (Real OPT-125M: vocab=50272, hidden=768)
    dummy_head = torch.empty(100, 768, dtype=torch.float32)
    torch.save(dummy_head, head_dir / "h4.pt")
    
    # Test it loads and generates
    llm = LLM(
        model="facebook/opt-125m",
        speculative_config={
            "method": "layer_skip",
            "layer_skip": 4,
            "num_speculative_tokens": 4,
            "lsq_head_path": str(head_dir),
            "draft_entropy_threshold": 2.0,
        }
    )
    
    out = llm.generate(["Hello"], SamplingParams(max_tokens=8))
    assert out[0].outputs[0].text
    assert len(out[0].outputs[0].token_ids) <= 8

def test_layer_skip_smoke_qwen(tmp_path):
    """Test with Qwen (if available)."""
    pytest.skip("Qwen model requires large download - manual test only")
    
    # Create dummy LSQ head for Qwen
    head_dir = tmp_path / "heads" 
    head_dir.mkdir()
    # Use small dummy head for CI - shape doesn't matter for smoke test
    # (Real Qwen-7B: vocab=152064, hidden=4096)
    dummy_head = torch.empty(100, 4096, dtype=torch.float32)
    torch.save(dummy_head, head_dir / "h16.pt")
    
    try:
        llm = LLM(
            model="Qwen/Qwen1.5-7B",
            speculative_config={
                "method": "layer_skip",
                "layer_skip": 16,  # Qwen-7B has 32 layers
                "num_speculative_tokens": 4,
                "lsq_head_path": str(head_dir),
                "draft_entropy_threshold": 2.0,
            }
        )
        out = llm.generate(["Shanghai is"], SamplingParams(max_tokens=8))
        assert out[0].outputs[0].text
    except Exception:
        pytest.skip("Qwen model not available")

def test_layer_skip_cli_args():
    """Test that CLI arguments are properly handled."""
    from vllm.utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import EngineArgs
    
    # Test CLI argument parsing
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    
    # Parse args with layer skip flags
    args = parser.parse_args([
        "--model", "facebook/opt-125m",
        "--speculative-layer-skip", "4", 
        "--lsq-head-path", "/tmp/heads"
    ])
    
    engine_args = EngineArgs(**vars(args))
    
    # Check that fields are set correctly
    assert engine_args.speculative_layer_skip == 4
    assert engine_args.lsq_head_path == "/tmp/heads"