import json
import inspect
from vllm.engine.arg_utils import EngineArgs


def _call_create_speculative_config(ea):
    # Create required configs
    from vllm.config import ModelConfig, ParallelConfig
    
    device_config = None
    if hasattr(ea, "create_device_config"):
        device_config = ea.create_device_config()
    
    # Create minimal model config for testing
    target_model_config = ModelConfig(
        model=ea.model,
        tokenizer=ea.model,  # Use model as tokenizer for simplicity
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="auto",
        seed=0,
    )
    
    target_parallel_config = ParallelConfig()

    # Respect the actual signature of your vLLM version
    sig = inspect.signature(ea.create_speculative_config)
    argmap = {
        "target_model_config": target_model_config,
        "target_parallel_config": target_parallel_config,
        "enable_chunked_prefill": False,
        "disable_log_stats": True,
    }
    
    args = []
    for name in sig.parameters:
        if name == "self":
            continue
        args.append(argmap.get(name, None))
    return ea.create_speculative_config(*args)


def test_engineargs_cli_merges_layer_skip_and_lsq(tmp_path):
    lsq_dir = tmp_path / "heads"
    lsq_dir.mkdir()

    ea = EngineArgs(
        model="facebook/opt-125m",
        speculative_config={"method": "layer_skip", "num_speculative_tokens": 5},  # Pass as dict, not JSON string
        speculative_layer_skip=6,
        lsq_head_path=str(lsq_dir),
        tensor_parallel_size=1,
    )

    cfg = _call_create_speculative_config(ea)

    assert cfg.method == "layer_skip"
    assert cfg.layer_skip == 6
    assert cfg.lsq_head_path == str(lsq_dir)
    assert cfg.num_speculative_tokens == 5