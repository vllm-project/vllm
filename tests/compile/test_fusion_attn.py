# SPDX-License-Identifier: Apache-2.0
import torch.nn

# yapf: disable
from tests.compile.backend import TestBackend
from vllm.attention import Attention, AttentionMetadata, get_attn_backend
from vllm.compilation.fusion_attn import AttnFusionPass
from vllm.compilation.fx_utils import find_auto_fn
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.platforms import current_platform

# yapf: enable


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


PassConfig = CompilationConfig.PassConfig


def test_attention_fusion(dist_init):
    torch.set_default_device("cuda")

    input = torch.rand(9, 4096)

    enable_attn_fusion = True
    pass_config = PassConfig(enable_attn_fusion=enable_attn_fusion,
                             enable_noop=True)
    compile_config = CompilationConfig(pass_config=pass_config)
    vllm_config = VllmConfig(compilation_config=compile_config)

    head_size = 32
    dtype = torch.bfloat16
    block_size = 16

    attn_backend = get_attn_backend(
        head_size,
        dtype,
        None,
        block_size,
        False,
    )
    impl = attn_backend.get_impl_cls()(0, 0, 0.0)
    assert impl.fused_output_quant_supported(current_platform.fp8_dtype(),
                                             True, False)
    # TODO use layerwise impls from layers instead

    # attn_meta_builder = attn_backend.get_builder_cls()()

    attn_metadata = AttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=5,
        num_decode_tokens=4,
        slot_mapping=torch.arange(0, 9),
        multi_modal_placeholder_index_maps=None,
        enable_kv_scales_calculation=False,
    )
    with set_current_vllm_config(vllm_config), set_forward_context(
            attn_metadata, vllm_config, num_tokens=9):
        # attention layers set the forward context on config
        model = TestModel(1)
        out = model(input)

        cc = vllm_config.compilation_config
        passes = []
        if enable_attn_fusion:
            # passes += [NoOpEliminationPass(cc.pass_config)] # TODO
            passes += [
                AttnFusionPass(cc.pass_config, cc.static_forward_context)
            ]
        inductor_backend = TestBackend(*passes)
        out2 = torch.compile(model, backend=inductor_backend)(input)

    torch.testing.assert_close(out, out2)

    if enable_attn_fusion:
        # TODO check fusion
        # TODO check if the backend supports it
        #  - need to make sure this always works so we can turn on by default
        # TODO use graph.find_nodes
        # TODO quant method for linear
        # TODO check that quant is gone
        # TODO fullgraph test
        node = find_auto_fn(inductor_backend.graph_post_pass.nodes,
                            torch.ops.vllm.unified_attention_with_output)
        assert node is not None
