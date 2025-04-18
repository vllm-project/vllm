# SPDX-License-Identifier: Apache-2.0

import torch
from torch import fx

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

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

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.static_fwd_ctx = config.compilation_config.static_forward_context

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        RESHAPE_OP = torch.ops.aten.reshape.default
        ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
        QUANT_OP = torch.ops._C.static_scaled_fp8_quant.default
        EMPTY_OP = torch.ops.aten.empty.memory_format

        # TODO(luka): move this to pass manager and be smarter with it
        from torch._inductor.fx_utils import FakeTensorUpdater
        fake_tensor_updater = FakeTensorUpdater(graph)
        nodes_to_process = []

        self.begin()
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
            attn_layer: Attention = self.static_fwd_ctx[layer_name]
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

            # 1. Set scale on unified_attn
            node.update_kwarg('output_scale', quant_node.kwargs['scale'])

            # 2. Rewrite the initial buf alloc dtype
            attn_output_buf.update_kwarg('dtype', quant_output.kwargs['dtype'])
            assert attn_output_buf.args == quant_output.args
            assert attn_output_buf.kwargs == quant_output.kwargs
            # meta value fixed by FakeTensorUpdater

            # 3. Update the autofunc node meta, as the FakeTensorUpdater does
            # not touch them. To get both the shape and type correct, we have
            # to create the type manually, which is a bit ugly.
            old_fake = node.meta['val'][1]
            new_fake_tensor = torch.empty_like(old_fake, dtype=quant_dtype)
            node.meta['val'] = (None, new_fake_tensor)

            # 4. Autofunc nodes don't get updated, so add the next node to the
            # update list. Also remove the meta value as the FakeTensorUpdater
            # has a bug where the type is not considered in FakeTensor equality.
            # No existing meta val forces a new one.
            nodes_to_process.append(attn_getitem)
            del attn_getitem.meta['val']

            # 5. Reshape output buffer back to 2d for scaled mm input
            with graph.inserting_after(attn_getitem):
                reshape_args = (attn_getitem, quant_output_shape)
                attn_new_output = graph.call_function(RESHAPE_OP, reshape_args)

            # 6. Rebind users of quant to use output of attention directly:
            find_getitem(quant_node, 1).replace_all_uses_with(attn_new_output)

            # Done, add to the count
            count += 1

        # remove the quant nodes
        graph.eliminate_dead_code()

        # Manually force processing of nodes
        for node in nodes_to_process:
            node_hash = fake_tensor_updater.hash_node(node)
            fake_tensor_updater.processed_hashes.remove(node_hash)

        # Update the fake tensor data
        fake_tensor_updater.incremental_update()

        logger.debug("fused quantization onto %s attention nodes", count)
        self.dump_graph(graph, "after_attn_fusion")
        self.end_and_log()
