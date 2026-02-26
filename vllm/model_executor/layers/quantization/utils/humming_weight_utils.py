# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from humming import dtypes
from humming.kernel import PackWeightKernel, UnpackWeightKernel
from humming.utils.weight import quantize_weight


def assert_shape_equal(expected_shape, actual_shape):
    assert expected_shape == actual_shape, (
        f"expected shape: {expected_shape}  actual shape: {actual_shape}"
    )


class HummingBaseWeightConverter:
    ckpt_weight_name: str = "weight"
    ckpt_weight_scale_name: str = "weight_scale"
    ckpt_zero_point_name: str = "zero_point"
    ckpt_global_scale_name: str = "global_scale"
    ckpt_bias_name: str = "bias"
    unused_names: tuple[str, ...] = ()

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.b_num_bits = self.quant_config.b_dtype.num_bits
        self.group_size = self.quant_config.weight_scale_group_size_k

    def _check_shape(
        self,
        tensor: torch.Tensor,
        dims: tuple[int, ...],
        num_experts: int | None = None,
        pack_dim: int | None = None,
        group_dim: int | None = None,
    ):
        if len(dims) == 1 and dims[0] == 1:
            assert tensor.ndim in [0, 1]
            assert tensor.nelement() == (num_experts or 1)
            return

        dims_list = list(dims)
        if pack_dim is not None:
            dims_list[pack_dim] = dims_list[pack_dim] * self.b_num_bits // 32
        if group_dim is not None:
            dims_list[group_dim] = self._get_num_groups(dims_list[group_dim])
        if num_experts is not None:
            dims_list = [num_experts] + dims_list
        assert_shape_equal(tuple(dims_list), tensor.shape)

    def _get_num_groups(self, shape_k: int):
        if self.group_size == 0:
            return 1
        else:
            assert shape_k % self.group_size == 0
            return shape_k // self.group_size

    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert tensor.dtype == torch.int32
        self._check_shape(tensor, (shape_n, shape_k), num_experts, pack_dim=1)
        return {"weight": tensor}

    def convert_weight_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if tensor.ndim == (1 if num_experts is None else 2):
            tensor = tensor.unsqueeze(-1)
        self._check_shape(tensor, (shape_n, shape_k), num_experts, group_dim=1)
        return {"weight_scale": tensor}

    def convert_zero_point(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if tensor.ndim == (1 if num_experts is None else 2):
            tensor = tensor.unsqueeze(-1)
        assert tensor.dtype == torch.int32
        self._check_shape(
            tensor, (shape_n, shape_k), num_experts, pack_dim=0, group_dim=1
        )
        return {"zero_point": tensor}

    def convert_global_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert tensor.dtype == torch.int32
        self._check_shape(tensor, (1,), num_experts)
        return {"global_scale": tensor}

    def convert_bias(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]
        self._check_shape(tensor, (shape_n,), num_experts)
        return {"bias": tensor}

    def convert(
        self,
        tensor: torch.Tensor,
        name: str,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ):
        convert_method_map = {
            "weight": self.convert_weight,
            "weight_scale": self.convert_weight_scale,
            "zero_point": self.convert_zero_point,
            "global_scale": self.convert_global_scale,
            "bias": self.convert_bias,
        }
        return convert_method_map[name](tensor, shape_n, shape_k, num_experts)

    def get_ckpt_name(self, param_name: str):
        return getattr(self, f"ckpt_{param_name}_name")

    def get_param_name(self, ckpt_name: str):
        for name in ["weight", "weight_scale", "zero_point", "global_scale", "bias"]:
            if self.get_ckpt_name(name) == ckpt_name:
                return name


class HummingUnquantizedWeightConverter(HummingBaseWeightConverter):
    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]
        self._check_shape(tensor, (shape_n, shape_k), num_experts)
        quanted_weight, weight_scale, zero_point, global_scale = quantize_weight(
            weight=tensor.cuda(),
            dtype=self.quant_config.b_dtype,
            scale_dtype=self.quant_config.bs_dtype,
            group_size=self.quant_config.weight_scale_group_size_k,
            has_dynamic_zp=self.quant_config.has_dynamic_zp,
            has_global_scale=self.quant_config.has_global_scale,
        )

        pack_kernel = PackWeightKernel(self.quant_config.b_dtype.num_bits)
        quanted_weight = pack_kernel(quanted_weight)

        if zero_point is not None:
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.squeeze().view(zero_point.shape)
            zero_point = pack_kernel(zero_point)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.squeeze().view(zero_point.shape)

        return {
            "weight": quanted_weight,
            "weight_scale": weight_scale,
            "zero_point": zero_point,
            "global_scale": global_scale,
        }


class HummingGPTQWeightConverter(HummingBaseWeightConverter):
    ckpt_weight_name: str = "qweight"
    ckpt_weight_scale_name: str = "scales"
    ckpt_zero_point_name: str = "qzeros"
    unused_names: tuple[str] = ("g_idx",)

    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.transpose(-1, -2).contiguous()
        return super().convert_weight(tensor, shape_n, shape_k, num_experts)

    def convert_weight_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.transpose(-1, -2).contiguous()
        return super().convert_weight_scale(tensor, shape_n, shape_k, num_experts)

    def convert_zero_point(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.transpose(-1, -2).contiguous()
        return super().convert_zero_point(tensor, shape_n, shape_k, num_experts)


class HummingAWQWeightConverter(HummingBaseWeightConverter):
    ckpt_weight_name: str = "qweight"
    ckpt_weight_scale_name: str = "scales"
    ckpt_zero_point_name: str = "qzeros"

    def __init__(self, quant_config):
        assert quant_config.b_dtype.num_bits == 4
        super().__init__(quant_config)

    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert tensor.dtype == torch.int32
        tensor = tensor.cuda()
        unpack_kernel = UnpackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = unpack_kernel(tensor)
        tensor = tensor.view(*tensor.shape[:-1], -1, 8)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        tensor = tensor.view(*tensor.shape[:-2], -1).transpose(-1, -2).contiguous()
        pack_kernel = PackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = pack_kernel(tensor)
        tensor = tensor.transpose(-1, -2).contiguous()
        return super().convert_weight(tensor, shape_n, shape_k, num_experts)

    def convert_weight_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.transpose(-1, -2).contiguous()
        return super().convert_weight_scale(tensor, shape_n, shape_k, num_experts)

    def convert_zero_point(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.cuda()
        unpack_kernel = UnpackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = unpack_kernel(tensor)
        tensor = tensor.view(*tensor.shape[:-1], -1, 8)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        pack_kernel = PackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = pack_kernel(tensor).squeeze(-1)
        tensor = tensor.transpose(-1, -2).contiguous()
        return super().convert_zero_point(tensor, shape_n, shape_k, num_experts)


class HummingFp8WeightConverter(HummingBaseWeightConverter):
    def __init__(self, quant_config):
        assert quant_config.b_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]
        super().__init__(quant_config)
        is_block_quant = self.quant_config.weight_scale_group_size_n > 1
        self.is_block_quant = is_block_quant
        if is_block_quant:
            self.ckpt_weight_scale_name = "weight_scale_inv"


class HummingModeloptWeightConverter(HummingBaseWeightConverter):
    ckpt_weight_scale_name: str = "weight_global_scale"

    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.view(torch.int32)
        return super().convert_weight(tensor, shape_n, shape_k, num_experts)

    def convert_weight_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.quant_config.bs_dtype == dtypes.float8e8m0:
            tensor = tensor.view(torch.float8_e8m0fnu)
        return super().convert_weight_scale(tensor, shape_n, shape_k, num_experts)


class HummingMxfp4WeightConverter(HummingBaseWeightConverter):
    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.view(torch.int32)
        return super().convert_weight(tensor, shape_n, shape_k, num_experts)

    def convert_weight_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.view(torch.float8_e8m0fnu)
        return super().convert_weight_scale(tensor, shape_n, shape_k, num_experts)


class HummingBitnetWeightConverter(HummingBaseWeightConverter):
    def convert_weight(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.cuda()  # (n // pack_factor, k)
        tensor = tensor.transpose(-1, -2).contiguous().view(torch.int32)
        unpack_kernel = UnpackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = unpack_kernel(tensor)
        tensor = tensor + 1  # (k, n)
        tensor = tensor.view(*tensor.shape[:-1], -1, 4).transpose(-1, -2).contiguous()
        tensor = tensor.view(*tensor.shape[:-2], -1)
        tensor = tensor.transpose(-1, -2).contiguous()  # (n, k)
        pack_kernel = PackWeightKernel(self.quant_config.b_dtype.num_bits)
        tensor = pack_kernel(tensor)
        return super().convert_weight(tensor, shape_n, shape_k, num_experts)

    def convert_weight_scale(
        self,
        tensor: torch.Tensor,
        shape_n: int,
        shape_k: int,
        num_experts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = 1 / tensor
        tensor = tensor.view(1) if num_experts is None else tensor.view(num_experts, 1)
        tensor = tensor.repeat_interleave(shape_n, -1)
        return super().convert_weight_scale(tensor, shape_n, shape_k, num_experts)


WEIGHT_CONVERTER_MAP: dict[str | None, type[HummingBaseWeightConverter]] = {
    "gptq": HummingGPTQWeightConverter,
    "awq": HummingAWQWeightConverter,
    "fp8": HummingFp8WeightConverter,
    "modelopt": HummingModeloptWeightConverter,
    "mxfp4": HummingMxfp4WeightConverter,
    "bitnet": HummingBitnetWeightConverter,
    None: HummingUnquantizedWeightConverter,
}
