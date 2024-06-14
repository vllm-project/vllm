import copy
import torch
import unittest.mock

from .code_cache import CodeCache
from .fusion import FusedOpGenerator, pointwise_fusion
from .register import SUPPORTED
from .rewrite_quantized_gemms import rewrite_quantized_gemms
from .utils import ModuleInputGenerator, graph_print_tabular, is_call, call_method_class, tag_side_effects

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

from vllm.logger import init_logger

import traceback

logger = init_logger(__name__)

###############################################################################
#
# Module partitioning
#
###############################################################################
"""
A callback for the fx CapabilityBasedPartitioner.  Nodes that are "supported"
are partitioned into new submodules.
"""
def is_node_supported(
    submodules: Mapping[str, torch.nn.Module],
    node: torch.fx.Node,
) -> bool:
    if is_call(node):
        fn_name = get_node_target(submodules, node)
        if not fn_name in SUPPORTED:
            return False

        if node.op == 'call_method':
            class_type = call_method_class(node)
            return any([issubclass(class_type, c) for c in SUPPORTED[fn_name]])
        else:
            return True
    else:
        return False


"""
Partition 'gm' into submodules based on the 'is_node_supported' callback.
Modules containing "supported" nodes will be optimized by the backend.
"""
def partition_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Tuple[torch.fx.GraphModule, List[Partition]]:
    support = create_op_support(is_node_supported)

    p = CapabilityBasedPartitioner(gm,
                                   support,
                                   allows_single_node_partition=False,
                                   non_compute_ops=None,
                                   allowed_single_node_partition_ops=None)
    parts = p.propose_partitions()
    part_gm = p.fuse_partitions(parts)

    return part_gm, parts


###############################################################################
#
# Inliner
#
###############################################################################
"""
Inline all submodules in 'mod' by running the tracer on them.
TBD
"""
def inline_submodules(mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    return mod


###############################################################################
#
# Backend
#
###############################################################################
"""
Run optimizer on the given module.
"""
def optimize(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> torch.fx.GraphModule:
    mod = rewrite_quantized_gemms(mod, example_inputs)
    mod = pointwise_fusion(cc, fgen, mod, example_inputs)
    # TODO: should we re-trace here to inline?  or will inductor handle it?
    # mod = inline_submodules(mod)
    return mod


def maybe_name(gm: torch.fx.GraphModule) -> Optional[str]:
    return gm.name if hasattr(gm, 'name') else None


"""
Compile a module with the given backend.
"""
def backend_compile(gm: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor],
                    backend: Optional[str] = 'inductor') -> Callable:
    if not backend:
        return gm.forward

    try:
        backend = lookup_backend(backend)
        logger.debug(f"attempting {backend} on {maybe_name(gm)}")  # TODO can have no name
        backend_compiled = backend(gm, example_inputs)
        if backend_compiled is not None:
            logger.debug(f"{backend} compiled {maybe_name(gm)}.")
            return backend_compiled
    except Exception as ex:
        logger.warning(f"backend_compile failed: {ex}")
        logger.warning(f"Trace: {traceback.format_tb(ex.__traceback__)}")
        pass

    return gm.forward


def node_in_module(n: torch.fx.Node, m: torch.fx.GraphModule) -> bool:
    # Note: this doesn't work: return n.graph.owning_module == m
    # Names should be unique, so this is ok
    return n.name in [nn.name for nn in m.graph.nodes]


def module_in_partitions(parts: List[Partition],
                         m: torch.fx.GraphModule) -> bool:
    for p in parts:
        if node_in_module(next(iter(p.nodes)), m):
            return True
    return False

#torch._dynamo.config.accumulated_cache_size_limit = 128

graph_counter = 0

class backend_class:
    """
    A custom backend for torch.compile.

    This backend works by partitioning the provided module into supported/unsupported
    sub-modules.  The supported submodules are passed to an optimizer and then compiled
    via an optional "final" backend.
    """

    # TODO: this probably needs additional context to avoid collisions, e.g.
    # module/model name.
    cc = CodeCache()

    def __init__(self, backend: Optional[str] = 'inductor'):
        self.backend = backend

    #
    # Remember that torch._dynamo.mark_dynamic(x, 0) can be used on example_inputs to mark dynamic
    # dimensions to prevent recompiles, these would go on the M dim (also kv_caches?)
    # Use TORCH_LOGS="recompiles" to see why things recompile
    #
    # TODO: if nothing is optimized, return original gm/gm.forward function
    #
    def __call__(self, gm: torch.fx.GraphModule,
                 example_inputs: List[torch.Tensor]) -> Callable:

        # Nop for baseline testing
        #print(f"Original module {gm} ({maybe_name(gm)}):\n{graph_print_tabular(gm.graph)}")
        #return backend_compile(gm, example_inputs, backend=self.backend)

        #logger.info("BACKEND")

        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._functorch.eager_transforms import functionalize
        from torch.fx import symbolic_trace

        # Must make a copy so that inductor backend doesn't choke.
        gm = copy.copy(gm)

        #print(f"Original module {gm}:\n{graph_print_tabular(gm.graph,'users',lambda n: n.users)}")
        #print(f"Original module {gm}:\n{graph_print_tabular(gm.graph)}")

        logger.debug(f"Original module {gm}:\n{graph_print_tabular(gm.graph)}")
        logger.debug(f"input_types: {[type(inp) for inp in example_inputs]}")

        if False:
            global graph_counter
            torch.fx.passes.graph_drawer.FxGraphDrawer(gm, "original").get_dot_graph().write_svg(f"original{graph_counter}.svg")
            torch.fx.passes.graph_drawer.FxGraphDrawer(gm, "original").get_dot_graph().write_png(f"original{graph_counter}.png")
            graph_counter = graph_counter + 1

        # See torch/_subclasses/function_tensor.py FunctionalTensor, FunctionalTensorMode
        # trace with FunctionTensor args???

        # See fx/passes/reinplace.py

        # See ./_functorch/_aot_autograd/dispatch_and_compile_graph.py:50 def aot_dispatch_base_graph
        # post_grad.py: decompose_auto_functionalized(graph):
        # from torch._higher_order_ops.auto_functionalize import auto_functionalized_dense

        tag_side_effects(gm.graph)

        gm.graph.eliminate_dead_code()

        if False:  # functionalize
            gm = make_fx(functionalize(gm,
                                       remove='mutations'))(*example_inputs)
            #print(
            #    f"Functionalized module {gm}:\n{graph_print_tabular(gm.graph,'users',lambda n: n.users)}"
            #)
            logger.debug(
                f"Functionalized module {gm}:\n{graph_print_tabular(gm.graph)}"
            )

        # TODO: store these in the root module state dictionary so that code for
        # all sub-modules is shared?  Or should these be globals?
        fgen = FusedOpGenerator()

        ShapeProp(gm).propagate(*example_inputs)

        #part_gm, parts = partition_graph(gm, example_inputs)
        part_gm = gm
        parts = [Partition(nodes=part_gm.graph.nodes)]

        logger.debug(f"Partitioned module: {part_gm.print_readable(False)}")
        logger.debug(f"parts: {parts}")

        # Get the current FakeTensorMode (there should be one since we are in
        # a backend).
        fake_mode = torch._guards.detect_fake_mode()

        # There should be an existing fake_mode but double check.
        assert fake_mode is not None

        # Ensure that 'allow_non_fake_inputs' is True so that original
        # 'example_inputs' can be used.
        # Using unittest.mock is borrowed from torch/_guards.py
        with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs",
                                        True):
            # Determine example inputs for submodules.
            # Note: static_shapes can be applied here if necessary.
            mig = ModuleInputGenerator(part_gm, fake_mode)
            mig.propagate(*example_inputs)

            part_gm = optimize(backend_class.cc, fgen, part_gm, example_inputs)

            # TODO: the partioner cannot handle side effect nodes?

            for name, m in part_gm.named_modules():
                if False and module_in_partitions(parts, m):
                    assert name in mig.module_args
                    module_inputs = mig.module_args[name][0]

                    # If one of the partitioned modules is called in multiple
                    # places, we skip it.  This should not happen though.
                    if not module_inputs:
                        logger.debug(f"SKIPPING {name}: multiple callers.")
                        continue

                    logger.debug(f"Optimizing {name}.")
                    m = optimize(backend_class.cc, fgen, m, module_inputs)
                    setattr(part_gm, name, m)

                    logger.debug(
                        f"Optimized {name}: {m.print_readable(False)}")

                    # TODO: don't really need to recompile if nothing was modified.
                    m.recompile()

                    if self.backend != None:
                        m.forward = backend_compile(m,
                                                    module_inputs,
                                                    backend=self.backend)

        if False:  # functionalize
            with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs",
                                            True):
                part_gm = torch.fx.passes.reinplace.reinplace(
                    part_gm, *example_inputs)

        part_gm.recompile()

        logger.debug(f"Final module: {part_gm.print_readable(False)}")

        # TODO: Add option for backend for the final graph?
        return part_gm.forward


"""
The default custom backend function for use with torch.compile.
"""
def backend(gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor]) -> Callable:
    return backend_class()(gm, example_inputs)


"""
Construct a custom torch.compile backend with optional 'final' backend for
optimized subgraphs. The default 'final' backend is the inductor. None can
be used instead to leave optimized subgraphs as interpreted.
"""
def make_backend(backend: Optional[str] = 'inductor') -> backend_class:
    return backend_class(backend)
