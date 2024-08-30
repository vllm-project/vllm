from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    MACHETE_SUPPORTED_GROUP_SIZES, check_machete_supports_shape,
    query_machete_supported_quant_types)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)

from .MPLinearKernel import *


class MacheteLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if c.act_reordering:
            return False, "Act reordering currently not supported by Machete"

        if c.zero_points:
            return False, "Zero points currently not supported by "\
                          " Compressed Tensors + Machete. (Kernel supports it"\
                          " but CompressedTensorsWNA16 does not so support has"\
                          " not been added to MacheteWNA16Kernel yet"

        if c.weight_type not in query_machete_supported_quant_types(
                c.zero_points):
            return False, f"Quant type ({c.weight_type}) not supported by "\
                           "Machete, supported types are: "\
                           f"{query_machete_supported_quant_types(c.zero_points)}"

        if c.group_size not in MACHETE_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({c.group_size}) not supported by "\
                            "Machete, supported group sizes are: "\
                            f"{MACHETE_SUPPORTED_GROUP_SIZES}"

        return check_machete_supports_shape(c.partition_weight_shape[0],
                                            c.partition_weight_shape[1])

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module):

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.machete_prepack_B(x.data.t().contiguous().t(),
                                           self.config.weight_type)
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        print(w_s)
        print(c.group_size)

        output = ops.machete_gemm(a=x_2d,
                                  b_q=w_q,
                                  b_type=c.weight_type,
                                  b_zeros=None,
                                  b_scales=w_s,
                                  b_group_size=c.group_size)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
