import torch
from humming import dtypes
from humming.kernel import PackWeightKernel, UnpackWeightKernel
from humming.utils.weight import quantize_weight


class HummingBaseWeightConverter:
    ckpt_weight_name = "weight"
    ckpt_weight_scale_name = "weight_scale"
    ckpt_zero_point_name = "zero_point"
    ckpt_global_scale_name = "global_scale"
    ckpt_bias_name = "bias"
    unused_names = ()

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def convert_weight(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight": tensor}

    def convert_weight_scale(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight_scale": tensor}

    def convert_zero_point(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"zero_point": tensor}

    def convert_global_scale(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"global_scale": tensor}

    def convert_bias(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"bias": tensor}

    def convert(self, tensor: torch.Tensor, name: str):
        convert_method_map = {
            "weight": self.convert_weight,
            "weight_scale": self.convert_weight_scale,
            "zero_point": self.convert_zero_point,
            "global_scale": self.convert_global_scale,
            "bias": self.convert_bias,
        }
        return convert_method_map[name](tensor)

    def get_ckpt_name(self, layer_name: str):
        return getattr(self, f"ckpt_{layer_name}_name")
    
    def get_param_name(self, ckpt_name: str):
        for name in ["weight", "weight_scale", "zero_point", "global_scale", "bias"]:
            if self.get_ckpt_name(name) == ckpt_name:
                return name


class HummingUnquantizedWeightConverter(HummingBaseWeightConverter):
    def convert_weight(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        quanted_weight, weight_scale, zero_point, global_scale = quantize_weight(
            weight=tensor,
            dtype=self.meta.b_dtype,
            scale_dtype=self.meta.bs_dtype,
            group_size=self.meta.weight_scale_group_size,
            has_dynamic_zp=self.meta.has_dynamic_zp,
            has_global_scale=self.meta.has_global_scale,
        )
        return {
            "weight": quanted_weight,
            "weight_scale": weight_scale,
            "zero_point": zero_point,
            "global_scale": global_scale,
        }


class HummingGPTQWeightConverter(HummingBaseWeightConverter):
    ckpt_weight_name = "qweight"
    ckpt_weight_scale_name = "scales"
    ckpt_zero_point_name = "qzeros"
    unused_names = ("g_idx",)

    def convert_weight(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight": tensor.transpose(-1, -2).contiguous()}

    def convert_weight_scale(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight_scale": tensor.transpose(-1, -2).contiguous()}

    def convert_zero_point(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"zero_point": tensor.transpose(-1, -2).contiguous()}


class HummingAWQWeightConverter(HummingBaseWeightConverter):
    ckpt_weight_name = "qweight"
    ckpt_weight_scale_name = "scales"
    ckpt_zero_point_name = "qzeros"

    def __init__(self, quant_config):
        assert quant_config.b_dtype.num_bits == 4
        super().__init__(quant_config)

    def convert_weight(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        assert tensor.dtype == torch.int32
        tensor = tensor.cuda()
        unpack_kernel = UnpackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = unpack_kernel(tensor)
        tensor = tensor.view(*tensor.shape[:-1], -1, 8)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        tensor = tensor.view(*tensor.shape[:-2], -1).transpose(-1, -2).contiguous()
        pack_kernel = PackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = pack_kernel(tensor)
        return {"weight": tensor}

    def convert_weight_scale(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight_scale": tensor.transpose(-1, -2).contiguous()}

    def convert_zero_point(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        tensor = tensor.cuda()
        unpack_kernel = UnpackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = unpack_kernel(tensor)
        tensor = tensor.view(*tensor.shape[:-1], -1, 8)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        pack_kernel = PackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = pack_kernel(tensor).squeeze(-1)
        return {"zero_point": tensor.transpose(-1, -2).contiguous()}


class HummingFp8WeightConverter(HummingBaseWeightConverter):
    def __init__(self, quant_config):
        assert quant_config.b_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]
        super().__init__(quant_config)
        is_block_quant = self.quant_config.weight_scale_group_size_n > 0
        self.is_block_quant = is_block_quant
        if is_block_quant:
            self.ckpt_weight_scale_name = "weight_scale_inv"


class HummingNvfp4WeightConverter(HummingBaseWeightConverter):
    ckpt_weight_name = "weight_packed"
    ckpt_weight_scale_name = "weight_global_scale"

    def convert_weight(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight": tensor.view(torch.int32)}

    def convert_weight_scale(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.is_block_quant:
            weight_scale_group_size_n = self.quant_config.weight_scale_group_size_n
            tensor = tensor.repeat_interleave(weight_scale_group_size_n, -1)
        return {"weight_scale": tensor}


class HummingMxfp4WeightConverter(HummingBaseWeightConverter):
    def convert_weight(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight": tensor.view(torch.int32)}

    def convert_weight_scale(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"weight_scale": tensor.view(torch.float8_e8m0fnu)}


WEIGHT_CONVERTER_MAP: dict[str | None, HummingBaseWeightConverter] = {
    "gptq": HummingGPTQWeightConverter,
    "awq": HummingAWQWeightConverter,
    "fp8": HummingFp8WeightConverter,
    "modelopt": HummingNvfp4WeightConverter,
    "mxfp4": HummingMxfp4WeightConverter,
    None: HummingUnquantizedWeightConverter,
}
