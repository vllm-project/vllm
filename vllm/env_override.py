# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger
from vllm.utils import is_torch_equal

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# see https://github.com/vllm-project/vllm/issues/10480
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1

# ===================================================
# torch 2.9 Inductor PythonWrapperCodegen monkeypatch
# ===================================================
# This change monkeypatches memory_plan_reuse in pytorch 2.9.0 to work around
# a test failure for test_multi_graph_piecewise_compile_outputs_equal.
# For more context, see https://github.com/pytorch/pytorch/pull/165514.


def memory_plan_reuse_patched(self):
    import torch._inductor.ir as ir
    from torch._inductor.codegen.wrapper import (
        EnterSubgraphLine,
        ExitSubgraphLine,
        MemoryPlanningLine,
        MemoryPlanningState,
        SubgraphPythonWrapperCodegen,
    )
    from torch._inductor.virtualized import V

    def get_output_names(graph_outputs) -> list[str]:
        import itertools

        names = []
        shape_counter = itertools.count(0)
        none_counter = itertools.count(0)
        for node in graph_outputs:
            if isinstance(node, ir.NoneAsConstantBuffer):
                names.append(f"{V.graph.name}_none{next(none_counter)}")
            elif isinstance(node, ir.ShapeAsConstantBuffer):
                names.append(f"{V.graph.name}_shape{next(shape_counter)}")
            else:
                names.append(node.get_name())
        return names

    if (
        isinstance(V.graph.wrapper_code, SubgraphPythonWrapperCodegen)
        and V.graph.wrapper_code.partition_signatures is not None
    ):
        out_names = get_output_names(
            V.graph.wrapper_code.partition_signatures.output_nodes
        )
    else:
        out_names = V.graph.get_output_names()

    while (
        self.lines
        and isinstance(self.lines[-1], MemoryPlanningLine)
        and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
    ):
        # these lines will be pointless
        self.lines.pop()

    # codegen allocations in two passes
    planning_states = [MemoryPlanningState()]
    past_planning_states = []
    for i in range(len(self.lines)):
        line = self.lines[i]
        if isinstance(line, MemoryPlanningLine):
            self.lines[i] = line.plan(planning_states[-1])
        elif isinstance(line, EnterSubgraphLine):
            planning_states.append(MemoryPlanningState())
        elif isinstance(line, ExitSubgraphLine):
            past_planning_states.append(planning_states.pop())
    past_planning_states.append(planning_states.pop())
    assert len(planning_states) == 0


# ========================================
# torch 2.9 Inductor Scheduler monkeypatch
# ========================================
# This change monkeypatches a function in Inductor to work around the following
# bug: https://github.com/vllm-project/vllm/issues/26678
#
# The bug occurs when `use_inductor_graph_partition` is turned on and there
# exists operators inside of `splitting_ops` that have an in-place mutation. In
# vllm, this specifically occurs on the operator
# vllm.unified_attention_with_output. In this case, inductor does not populate
# the inductor IR's `origin_node` field, causing an assertion error when trying
# to access the node's `origin_node` field.
#
# So, we will monkeypatch torch._inductor.scheduler.Scheduler.should_partition
# so that it does not access the inductor IR node's `origin_node` field and just
# returns True if a node is registered as having a custom partition function.
# This is ok for now since vllm's implementation of the custom partition
# functions just return True.
# ========================================


def should_partition_patched(self, node, should_log: bool = False) -> bool:
    # This is a patched version of
    # torch._inductor.scheduler.Scheduler.should_partition that modifies
    # the following piece of code so that we always return True:
    # https://github.com/pytorch/pytorch/blob/ecb53078faf86ca1b33277df33b82985675bb011/torch/_inductor/scheduler.py#L4712-L4724
    """Return True if we should partition the inductor graph on this node"""

    import torch._inductor.ir as ir
    from torch._inductor.scheduler import (
        BaseSchedulerNode,
        FusedSchedulerNode,
        _custom_should_partition_fns,
    )
    from torch._inductor.utils import (
        _unstable_customized_partition_wrapper,
        is_cudagraph_unsafe_op,
        maybe_log_cudagraph_partition,
    )

    # Allow users to manually specify if a node should be partitioned
    # Can only do this for FallbackKernels
    ir_node = node.node
    if isinstance(ir_node, ir.FallbackKernel):
        operator = ir_node.op_overload
        if operator is not None and operator in _custom_should_partition_fns:
            return True

    # When not using cudagraphs, keep all kernels in the `call` function
    # instead of graph partition functions, since graph partition only brings
    # benefit to cudagraph
    if (
        not torch._inductor.config.triton.cudagraphs
        and _unstable_customized_partition_wrapper.wrapper is None
    ):
        return True

    # avoid duplicating logs when should_partition is called multiple times
    # on the same node
    def noop_log(msg: str, node: BaseSchedulerNode | None) -> None:
        return

    log_partition_reason = maybe_log_cudagraph_partition if should_log else noop_log

    if isinstance(node, FusedSchedulerNode):
        return any(self.should_partition(snode) for snode in node.snodes)

    assert node.node is not None

    if not node.is_gpu():
        log_partition_reason("non gpu ops", node=node)

        return True

    if isinstance(node.node, ir.DeviceCopy):
        log_partition_reason("DeviceCopy ops", node=node)
        return True

    if isinstance(node.node, ir.Conditional):
        log_partition_reason("Conditional ops", node=node)
        return True

    if getattr(node.node, "unbacked_bindings", None):
        log_partition_reason("unbacked binding ops", node=node)
        return True

    if is_cudagraph_unsafe_op(node.node):
        log_partition_reason("CUDAGraph-unsafe custom ops", node=node)
        return True

    return False


def _update_scheduler_patched(self) -> None:
    # Copied from torch._inductor.graph.GrahLowering._update_scheduler. Patches
    # this method so that we can patch Scheduler.should_partition with the
    # function above
    """
    (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
    files should be generated (to avoid biasing any benchmarks and pessimizing
    fusion decisions).
    """
    import torch._inductor.config as config
    from torch._inductor.scheduler import Scheduler

    Scheduler.should_partition = should_partition_patched

    with config.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


if is_torch_equal("2.9.0"):
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.graph import GraphLowering

    PythonWrapperCodegen.memory_plan_reuse = memory_plan_reuse_patched
    GraphLowering._update_scheduler = _update_scheduler_patched
