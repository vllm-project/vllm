import torch
from torch.nn import Module
from .torch_compile import kMoEFinalizeARResidualRMSNorm

class MoEModel(Module):
    def __init__(self, num_experts, hidden_size):
        super(MoEModel, self).__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden_size))
        self.bias = torch.nn.Parameter(torch.randn(num_experts, hidden_size))
        self.residual = torch.nn.Parameter(torch.randn(hidden_size))
        self.allreduce = torch.distributed.all_reduce
        self.rms_norm = torch.nn.RMSNorm(hidden_size)

    def forward(self, input):
        output = kMoEFinalizeARResidualRMSNorm(input, self.weight, self.bias, self.residual, self.allreduce, self.rms_norm)
        return output