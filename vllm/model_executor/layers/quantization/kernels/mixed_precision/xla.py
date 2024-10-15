from typing import Optional, Tuple

import torch

from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

XLA_SUPPORTED_QUANT_TYPES = [scalar_types.uint8b128]
XLA_SUPPORTED_GROUP_SIZES = [-1]


class XLAMixedPrecisionLinearKernel(MPLinearKernel):
    """
    XLAMixedPrecisionLinearKernel: WNA16 for TPU
    
    Kernel definition:
        - https://github.com/pytorch/xla/blob/v2.5.0-rc9/torch_xla/experimental/xla_quantized_matmul.py#L78

    Supported:
        - w8a16 symmetric channelwise

    Currently unsupported:
        - w8a16
        - w8a16 grouped
        - w4a16 
        - asymmetric
        - activation_reordering
    """

    @classmethod
    def get_min_capability(cls) -> Optional[int]:
        return None

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:

        if c.zero_points:
            return False, "Zero points currently not supported by XLA"

        if c.weight_type not in XLA_SUPPORTED_QUANT_TYPES:
            return False, f"Quant type ({c.weight_type}) not supported by XLA"\
                          f" , supported types are: {XLA_SUPPORTED_QUANT_TYPES}"

        if c.group_size not in XLA_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({c.group_size}) not supported by "\
                            "XLA, supported group sizes are: "\
                            f"{XLA_SUPPORTED_GROUP_SIZES}"

        if c.has_g_idx:
            return False, "Activation reordering is not supported by XLA"

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        def transform_w_int8(x):
            # reinterpet_cast the packed int32 weights as int8
            # convert to [int, out] -> [out, int]
            return x.view(dtype=torch.int8).t()

        def transform_s_channelwise(x):
            # convert to [out]
            return x.squeeze(-1).to(torch.bfloat16)

        self._transform_param(layer, self.w_q_name, transform_w_int8)
        self._transform_param(layer, self.w_s_name, transform_s_channelwise)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)
        assert w_zp is None and w_gidx is None

        #weight_scale = weight_scale.squeeze(-1).to(torch.bfloat16)
        import torch_xla.experimental.xla_quantized_matmul  # noqa: F401
        return torch.ops.xla.quantized_matmul(x,
                                              w_q,
                                              w_s,
                                              quantize_activation=False)
