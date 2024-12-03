from typing import Optional

from transformers import LlamaConfig


class LlamaSwiftKVConfig(LlamaConfig):
    """
    Args:
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
        last_key_value_heads (int, optional):
            The number of heads in the last layer that have keys and values.
            If None, the number of heads in the last key-value layer is equal
            to the number of heads in all the other key-value layers.
    """

    model_type = "llama_swiftkv"

    def __init__(
        self,
        swiftkv: bool = False,
        num_key_value_layers: Optional[int] = None,
        key_value_group_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiftkv = swiftkv
        self.num_key_value_layers = num_key_value_layers or self.num_hidden_layers
        self.key_value_group_size = key_value_group_size or 1
        assert (self.num_hidden_layers - self.num_key_value_layers) % self.key_value_group_size == 0
