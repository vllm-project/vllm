# SPDX-License-Identifier: Apache-2.0
import torch.nn

from vllm.attention import Attention
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)


class TestLayer(torch.nn.Module):

    def __init__(self,
                 head_size=32,
                 hidden_dim=4096,
                 num_heads=128,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv_size = self.num_heads * self.head_size
        self.hidden_dim = hidden_dim
        self.attn = Attention(self.num_heads, self.head_size, 0.5)
        self.qkv_proj = QKVParallelLinear(self.hidden_dim,
                                          self.head_size,
                                          self.num_heads,
                                          bias=False,
                                          return_bias=False)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_size,
                                        self.hidden_dim,
                                        bias=False,
                                        return_bias=False)

    def forward(self, input_: torch.Tensor):
        qkv = self.qkv_proj(input_)
        q, k, v = qkv.split([self.qkv_size] * 3, dim=-1)
        out = self.attn(q, k, v)
        return self.o_proj(out)


class TestModel(torch.nn.Module):

    def __init__(self,
                 num_layers: int,
                 head_size=32,
                 hidden_dim=4096,
                 num_heads=128,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.Sequential(
            *(TestLayer(head_size, hidden_dim, num_heads)
              for _ in range(num_layers)))

    def forward(self, input: torch.Tensor):
        return self.layers(input)


def test_attention_fusion(dist_init):
    torch.set_default_device("cuda")

    model = TestModel(1)
    input = torch.rand(5, 4096)

    with set_forward_context():
        out = model(input)
        out2 = torch.compile(model)(input)

        torch.testing.assert_close(out, out2)
