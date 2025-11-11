from transformers import PretrainedConfig

from vllm.transformers_utils.config import is_interleaved


def test_is_interleaved():
    # interleaved full+chunked e.g. Llama4
    chunked_layers_config = {
        "layer_types": [
            "chunked_attention",
            "chunked_attention",
            "chunked_attention",
            "full_attention",
            "chunked_attention",
            "chunked_attention",
            "chunked_attention",
            "full_attention",
            "chunked_attention",
            "chunked_attention",
            "chunked_attention",
            "full_attention",
        ]
    }
    assert is_interleaved(PretrainedConfig(**chunked_layers_config))

    # interleaved full+linear e.g. MiniMax-M1
    linear_layers_config = {
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    }
    assert is_interleaved(PretrainedConfig(**linear_layers_config))
