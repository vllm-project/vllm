# SPDX-License-Identifier: Apache-2.0

import torch
from torch import fx

from vllm.logger import init_logger

from ..attention import Attention
from ..config import CompilationConfig
from .fx_utils import find_getitem, get_only_user, is_auto_func, is_func
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class AttnFusionPass(VllmInductorPass):
    """
    - check if output is quantized
    - set the scale on the attention backend
      - lookup via layer name and config (need config)
    - remap the out from scaled_fp8_quant to attention directly
    - remap the meta vals

    TODO with pattern_matcher:
     - hope the attention layers are available so we can extract dims
     - otherwise wildcard the sizes?
    """

    def __init__(self, config: CompilationConfig.PassConfig,
                 context: dict[str, Attention]):
        super().__init__(config)

        # TODO(luka): need to improve config passing
        self.static_fwd_context = context

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        RESHAPE_OP = torch.ops.aten.reshape.default
        ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
        QUANT_OP = torch.ops._C.static_scaled_fp8_quant.default
        EMPTY_OP = torch.ops.aten.empty.memory_format

        self.dump_graph(graph, "before_attn_fusion")
        count = 0
        for node in graph.nodes:
            # Find the unified attention nodes
            if not is_auto_func(node, ATTN_OP):
                continue

            attn_getitem = find_getitem(node, 1)

            # Output view might be reshaped, skip reshapes until the real use
            output_view = attn_getitem

            while is_func(get_only_user(output_view), RESHAPE_OP):
                output_view = get_only_user(output_view)

            # If this is not a quant node, exit
            if not is_auto_func(get_only_user(output_view), QUANT_OP):
                continue

            quant_node: fx.Node = get_only_user(output_view)
            quant_output: fx.Node = quant_node.kwargs["result"]
            assert is_func(quant_output, EMPTY_OP)
            quant_output_shape = quant_output.args[0]

            # Extract pertinent attention information,
            # assuming the output is allocated 2d and reshaped
            layer_name = node.kwargs["layer_name"]
            attn_output_view = node.kwargs["output"]
            assert is_func(attn_output_view, RESHAPE_OP)
            attn_output_buf = attn_output_view.args[0]
            assert is_func(attn_output_buf, EMPTY_OP)

            # Check that the attention layer impl supports quant fusion
            attn_layer: Attention = self.static_fwd_context[layer_name]
            quant_dtype = quant_output.kwargs['dtype']
            # hardcoded because we only match the op for static/per-tensor
            # (static_scaled_fp8_quant) for now
            static, per_token = True, False
            if not attn_layer.impl.fused_output_quant_supported(
                    quant_dtype, static=static, per_token=per_token):
                continue

            # The output shapes and dtypes are:
            # - attn_view: bf16[s0, n_heads, head_size]
            # - attn_buf: bf16[s0, n_heads * head_size]
            # - quant: fp8[s0, n_heads * head_size]

            # START REWRITE
            # After this point, all operations must succeed for the graph to
            # remain valid. That includes modifications to attention layers.

            # 1. Set scale on Attention layer object
            # TODO
            attn_layer._o_scale = quant_node.kwargs['scale']
            print(attn_layer)

            # 2. Rewrite the graph
            # 2.a Rewrite the initial buf alloc dtype
            attn_output_buf.update_kwarg('dtype', quant_output.kwargs['dtype'])
            assert attn_output_buf.args == quant_output.args
            assert attn_output_buf.kwargs == quant_output.kwargs
            assert quant_output.meta['val'].dtype == quant_dtype
            attn_output_buf.meta['val'] = quant_output.meta['val']

            # 2.b Propagate the dtype TODO
            # attn_output_view.meta['val'].dtype = quant_dtype
            # node.meta['val'][1].dtype = quant_dtype
            # attn_getitem.meta['val'].dtype = quant_dtype

            # 2.d Reshape output buffer back to 2d for scaled mm input
            with graph.inserting_after(attn_getitem):
                reshape_args = (attn_getitem, quant_output_shape)
                attn_new_output = graph.call_function(RESHAPE_OP, reshape_args)

            # 2.e Rebind users of quant to use output of attention directly:
            find_getitem(quant_node, 1).replace_all_uses_with(attn_new_output)

            count += 1

        logger.debug("fused quantization onto %s attention nodes", count)
        self.dump_graph(graph, "after_attn_fusion")

    def error_out(self, message: str):
        # TODO(luka) this could be an error?
        logger.warning(
            "Unexpected fx.Graph state while fusing quant onto attention. "
            "If you expect this to work, please submit a bug report to "
            "https://github.com/vllm-project/vllm/issues/new/choose. "
            "Reason: %s.", message)
