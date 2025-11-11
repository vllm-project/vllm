from transformers import PretrainedConfig

from vllm.transformers_utils.config import is_interleaved


def test_is_interleaved():
    # interleaved full+sliding
    sliding_window_layers_config = {
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ]
    }
    assert is_interleaved(PretrainedConfig(**sliding_window_layers_config))

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

    # no interleaved layers
    full_attn_layers_config = {
        "layer_types": [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ]
    }
    assert not is_interleaved(PretrainedConfig(**full_attn_layers_config))
