# adapt from https://github.com/Guangxuan-Xiao/torch-int
import torch
from vllm import i8gemm
from .quantization import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)
from vllm.ftgemm import FTGEMM
ftgemm = FTGEMM()


class W8A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

    def _apply(self, fn):
        super()._apply(fn)
        self.a = self.a.cpu()
        self.b = self.b.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = i8gemm.linear_a8_w8_b8_o8(x, self.weight, self.bias,
                               self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        # FIXME: Just adapt to ParallelLinears' output
        return y, None

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O8Linear(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        # int8_bias, bias_scale should be 0, 0.0
        mockbias = torch.zeros((1, module.out_features), dtype=torch.int8, requires_grad=False)
        int8_bias, bias_scale = quantize_per_tensor_absmax(mockbias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8B8O8LinearWithSFactor(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0, inscale=1.0, ouscale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))
        self.register_buffer('inscale', torch.tensor(inscale))
        self.register_buffer('ouscale', torch.tensor(ouscale))

    def _apply(self, fn):
        super()._apply(fn)
        self.a = self.a.cpu()
        self.b = self.b.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x = (x / self.inscale).round().clamp(-128, 127).to(torch.int8)
        y = i8gemm.linear_a8_w8_b8_o8(x, self.weight, self.bias,
                               self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y, None

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O8LinearWithSFactor(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        mockbias = torch.zeros((1, module.out_features), dtype=torch.int8, requires_grad=False)
        int8_bias, bias_scale = quantize_per_tensor_absmax(mockbias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        int8_module.inscale = input_scale
        int8_module.ouscale = output_scale
        return int8_module


class W8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.a = self.a.cpu()
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        self.bias = self.bias.to(torch.float32)
        y = i8gemm.linear_a8_w8_bfp32_ofp32(
            x, self.weight, self.bias, self.a.item(), 1)
        y = y.view(*x_shape[:-1], -1)
        return y, None

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32Linear(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        mockbias = torch.zeros((1, module.out_features), dtype=torch.float, requires_grad=False)
        int8_module.bias = mockbias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module


class W8A8BFP32OFP32LinearWithSFactor(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, inscale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('inscale', torch.tensor(inscale))


    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.a = self.a.cpu()
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        self.a = self.a.to(*args, **kwargs)
        self.a = self.a.to(torch.float32)
        self.inscale = self.inscale.to(*args, **kwargs)
        self.inscale = self.inscale.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # quant activation
        x = (x / self.inscale).round().clamp(-128, 127).to(torch.int8)
        self.bias = self.bias.to(torch.float32)
        y = i8gemm.linear_a8_w8_bfp32_ofp32(
            x, self.weight, self.bias, self.a.item(), 1)
        y = y.view(*x_shape[:-1], -1)
        return y, None

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32LinearWithSFactor(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        mockbias = torch.zeros((1, module.out_features), dtype=torch.float, requires_grad=False)
        int8_module.bias = mockbias.to(torch.float32)
        int8_module.a = alpha
        int8_module.inscale = torch.tensor(input_scale)
        return int8_module

# use ftgemm a8w8o8
class W8A8BFP32OFP32LinearWithSFactorCublas(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, inscale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('inscale', torch.tensor(inscale))


    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.a = self.a.cpu()
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        self.a = self.a.to(*args, **kwargs)
        self.a = self.a.to(torch.float32)
        self.inscale = self.inscale.to(*args, **kwargs)
        self.inscale = self.inscale.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # quant activation
        x = (x / self.inscale).clamp(-128, 127).to(torch.int8)
        # self.bias = self.bias.to(torch.float32)
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int8, device=x.device)
        ftgemm.linear_a8_w8_o8_(x, self.weight, y, self.a.item())
        # y = i8gemm.linear_a8_w8_bfp32_ofp32(
        #     x, self.weight, self.bias, self.a.item(), 1)
        # int8 to float32
        y = y.to(torch.float32)
        y = y.view(*x_shape[:-1], -1)
        return y, None

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32LinearWithSFactorCublas(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        mockbias = torch.zeros((1, module.out_features), dtype=torch.float, requires_grad=False)
        int8_module.bias = mockbias.to(torch.float32)
        int8_module.a = alpha
        int8_module.inscale = torch.tensor(input_scale)
        return int8_module

# use ftgemm a8w8o8
class W8A8BFP32OFP32LinearCublas(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.a = self.a.cpu()
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # self.bias = self.bias.to(torch.float32)
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int8, device=x.device)
        ftgemm.linear_a8_w8_o8_(x, self.weight, y, self.a.item())
        # y = i8gemm.linear_a8_w8_bfp32_ofp32(
        #     x, self.weight, self.bias, self.a.item(), 1)
        # int8 to float32
        y = y.to(torch.float32)
        y = y.view(*x_shape[:-1], -1)
        return y, None

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32LinearCublas(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        mockbias = torch.zeros((1, module.out_features), dtype=torch.float, requires_grad=False)
        int8_module.bias = mockbias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module