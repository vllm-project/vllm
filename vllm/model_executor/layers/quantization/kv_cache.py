from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase, QuantizationConfig
import torch

class BaseKVCacheMethod(QuantizeMethodBase):

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka k_scale and v_scale) for an attention layer.
        The scales will be used to:
         - quantize kv_cache entries before saving them to the cache
         - dequantize kv_cache entries before fetching them from the cache
        Args:
            layer: The layer that is using the QuantizeMethodBase factory.
        """
        # Initialize the KV cache scales to 1.0 as the default value.
        # If the k_scale/v_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        layer.k_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # If the kv-cache dtype is auto, we enforce the kv-scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        layer.kv_cache_dtype = None
        if layer.kv_cache_dtype != "auto":
            for scale_name, scale in (("k_scale", layer.k_scale), ("v_scale", layer.v_scale)):
                scale = getattr(layer, scale_name).to("cpu").tolist()
                if not isinstance(scale, float):
                    # TODO: Once we retire KV Cache quantization using
                    # FP8 for compressed-tensors, we should get rid of this check
                    # here
                    raise ValueError("Only support per-tensor scaling factor "
                                    "for compressed-tensors KV cache")
                setattr(layer, f'_{scale_name}', scale)
        del layer.k_scale
        del layer.v_scale