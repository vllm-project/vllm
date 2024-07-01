from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase, QuantizationConfig
import torch
from vllm.utils import print_warning_once


class BaseKVCacheMethod(QuantizeMethodBase):

    def __init__(self,
                 quant_config: QuantizationConfig,
                 needs_scale_merging: bool = False):
        self.quant_config = quant_config
        # scale merging needs to be True if we expect to load
        # k_scale and v_scale first and then compute
        # kv_scale = max(k_scale, v_scale)
        self._needs_scale_merging = needs_scale_merging

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka kv_scale) for an attention layer.
        The scales will be used to:
         - quantize kv_cache entries before saving them to the cache
         - dequantize kv_cache entries before fetching them from the cache
        Args:
            layer: The layer that is using the QuantizeMethodBase factory.
        """
        # Initialize the KV cache scale to 1.0 as the default value.
        # If the kv_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        if self._needs_scale_merging:
            layer._k_scale = torch.nn.Parameter(torch.tensor(1.0),
                                                requires_grad=False)
            layer._v_scale = torch.nn.Parameter(torch.tensor(1.0),
                                                requires_grad=False)
            return

        layer.kv_scale = torch.nn.Parameter(torch.tensor(1.0),
                                            requires_grad=False)

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(
            f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._needs_scale_merging:
            layer.kv_scale = torch.nn.Parameter(torch.max(
                layer._v_scale, layer._k_scale),
                                                requires_grad=False)
            del layer._v_scale
            del layer._k_scale

        # If the kv-cache dtype is auto, we enforce the kv-scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        if layer.kv_cache_dtype != "auto":
            kv_scale = layer.kv_scale.to("cpu").tolist()
            if not isinstance(kv_scale, float):
                raise ValueError(
                    "Currently we only support per-tensor scaling factor (float)"
                )
            layer._kv_scale = kv_scale
            # TODO: We should potentially move this check elsewhere, to discuss
            if layer._kv_scale == 1.0 and "e5m2" not in layer.kv_cache_dtype:
                print_warning_once(
                    "Using KV cache scaling factor 1.0 for fp8_e4m3. This may "
                    "cause accuracy issues. Please make sure kv-cache scaling "
                    "factor is available in the fp8 checkpoint.")
        del layer.kv_scale
