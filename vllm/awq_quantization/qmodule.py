# adapted from llm-awq: https://github.com/mit-han-lab/llm-awq

import torch
import torch.nn as nn

try:
    import awq_inference_engine  # with CUDA kernels
except ImportError as ex:
    raise ImportError(
        "Unable to import awq_inference_engine: run setup.py"
        " to install AWQ CUDA kernels")


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(
            self,
            w_bit,
            group_size,
            in_features,
            out_features,
            bias,
            dev
        ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        qweight_buffer = torch.empty(
            (in_features, out_features // (32 // self.w_bit)),
            dtype=torch.int32,
            device=dev
        )
        self.register_buffer("qweight", qweight_buffer)

        qzeros_buffer = torch.empty(
            (
                in_features // self.group_size,
                out_features // (32 // self.w_bit)
            ),
            dtype=torch.int32,
            device=dev
        )
        self.register_buffer("qzeros", qzeros_buffer)

        scales_buffer = torch.empty(
            (in_features // self.group_size, out_features),
            dtype=torch.float16,
            device=dev
        )
        self.register_buffer("scales", scales_buffer)

        if bias:
            bias_buffer = torch.empty(
                (out_features),
                dtype=torch.float16,
                device=dev
            )
            self.register_buffer("bias", bias_buffer)
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )

        out = awq_inference_engine.gemm_forward_cuda(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            8
        )

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        str_repr = "in_features={}, out_features={}, " \
                   "bias={}, w_bit={}, group_size={}"
        return str_repr.format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.w_bit,
            self.group_size
        )
