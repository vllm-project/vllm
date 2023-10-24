# adapt from https://github.com/Guangxuan-Xiao/torch-int
import torch
from vllm.i8cugemm import I8CUGEMM
i8cugemm = I8CUGEMM()

class W8A8OFP32LinearWithSFactorCublas(torch.nn.Module):
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
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y * self.a.item()
        y = y.view(*x_shape[:-1], -1)
        return y, None


class W8A8O32LinearCublas(torch.nn.Module):
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
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y * self.a.item()
        y = y.view(*x_shape[:-1], -1)
        return y, None


class W8A8OFP32LinearWithSFactorCublasNoQuant(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        # self.register_buffer('a', torch.tensor(alpha))
        # self.register_buffer('inscale', torch.tensor(inscale))


    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        # self.a = self.a.cpu()
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        # self.a = self.a.to(*args, **kwargs)
        # self.a = self.a.to(torch.float32)
        # self.inscale = self.inscale.to(*args, **kwargs)
        # self.inscale = self.inscale.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # quant activation
        # x = (x / self.inscale).round().clamp(-128, 127).to(torch.int8)
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        # y = y * self.a.item()
        y = y.view(*x_shape[:-1], -1)
        return y, None


class W8A8O32LinearCublasNoDequant(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        # self.register_buffer('a', torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        # self.a = self.a.cpu()
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
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        # y = y * self.a.item()
        y = y.view(*x_shape[:-1], -1)
        return y, None
    

class W8A8O32Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
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
        y = torch.empty((x.shape[0], self.out_features), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y, None