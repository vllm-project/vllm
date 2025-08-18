import json
from vllm.engine.arg_utils import EngineArgs


def test_engineargs_cli_merges_layer_skip_and_lsq(tmp_path):
    lsq_dir = tmp_path / "heads"
    lsq_dir.mkdir()

    ea = EngineArgs(
        model="facebook/opt-125m",
        speculative_config=json.dumps({"method": "layer_skip", "num_speculative_tokens": 5}),
        speculative_layer_skip=6,
        lsq_head_path=str(lsq_dir),
        tensor_parallel_size=1,
    )
    # Ensure device_config is not None to prevent runtime errors
    if ea.device_config is None:
        ea.device_config = ea.create_device_config()
    
    cfg = ea.create_speculative_config(
        ea.model,
        ea.tokenizer,
        ea.tokenizer_mode,
        ea.trust_remote_code,
        ea.download_dir,
        ea.load_format,
        ea.dtype,
        ea.quantization_param,
        ea.max_context_len_to_capture,
        ea.device_config,
    )
    assert cfg.method == "layer_skip"
    assert cfg.layer_skip == 6
    assert cfg.lsq_head_path == str(lsq_dir)
    assert cfg.num_speculative_tokens == 5