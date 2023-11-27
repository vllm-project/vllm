# Adapted from
# https://github.com/foundation-model-stack/foundation-model-stack/blob/14dcaf0c578532639bd51b8be5809430b31872f4/fms/distributed/tensorparallel.py

from typing import List

import torch
import torch._inductor.codegen.wrapper as inductor_wrapper
import torch._inductor.ir as inductor_ir
import torch._inductor.lowering as inductor_lowering
import torch.distributed._functional_collectives as distfunc


def tensor_model_parallel_all_reduce(x: torch.Tensor,
                                     reduce_op: str = "sum") -> torch.Tensor:
    # NOTE(woosuk): get_tensor_model_parallel_world_size is not compatible with
    # torch.compile because it uses the _TENSOR_MODEL_PARALLEL_GROUP global
    # variable.
    tp_world_size = torch.distributed.get_world_size()
    if tp_world_size == 1:
        return x
    return distfunc.all_reduce(x, reduce_op, list(range(tp_world_size)))


# Fix #1 is porting the code changes in https://github.com/pytorch/pytorch/pull/108811
@classmethod
def wait_create(cls, collective_op: "inductor_ir.TensorBox"):
    collective_op.decide_layout()
    return inductor_ir.Wait(
        layout=inductor_ir.AliasedLayout(collective_op),
        inputs=[collective_op],
    )


inductor_ir.Wait.create = wait_create

inductor_ir.AllReduce.get_mutation_names = lambda self: [
    self.inputs[0].get_name()
]


@classmethod
def all_reduce_create(
    cls,
    x: "inductor_ir.TensorBox",
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    inplace_inputs = cls.wrap_inputs_as_inplace([x])
    layout = inductor_ir.MutationLayout(inplace_inputs[0])

    _ = inductor_ir.AllReduce(
        layout=layout,
        inputs=inplace_inputs,
        constant_args=[tag, ranks, group_size],
        reduce_op=reduce_op,
    )
    return inplace_inputs[0]


inductor_ir.AllReduce.create = all_reduce_create


def wcg_codegen_free(self, buffer):
    name = buffer.get_name()

    # can be freed but not reused
    # TODO: Port this one-line fix to PyTorch
    if isinstance(buffer, (inductor_ir.InputBuffer, inductor_ir.OutputBuffer)):
        self.writeline(self.make_buffer_free(buffer))
        return

    if not self.can_reuse(buffer):
        return
    self.freed.add(name)

    layout = buffer.get_layout()
    if isinstance(layout,
                  (inductor_ir.AliasedLayout, inductor_ir.MultiOutputLayout)):
        self.writeline(self.make_buffer_free(buffer))
        return

    self.writeline(inductor_wrapper.FreeIfNotReusedLine(self, buffer))


inductor_wrapper.WrapperCodeGen.codegen_free = wcg_codegen_free
# End of fix #1

# Fix #3: Avoid recompiles on batch size for embedding + TP
# (until https://github.com/pytorch/pytorch/pull/109561 lands)
for overload in torch.ops.c10d_functional.all_gather_into_tensor.overloads():
    other_fn = getattr(torch.ops.c10d_functional.all_gather_into_tensor,
                       overload)
    if other_fn in inductor_lowering.lowerings:
        del inductor_lowering.lowerings[other_fn]


@inductor_lowering.register_lowering(
    torch.ops.c10d_functional.all_gather_into_tensor)
def all_gather_into_tensor(shard, tag, ranks, group_size):
    return inductor_ir.TensorBox.create(
        inductor_ir.AllGatherIntoTensor.create(
            inductor_ir.ExternKernel.require_contiguous(shard), tag, ranks,
            group_size))


def tensor_model_parallel_all_gather(x: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return x

    if dim < 0:
        dim = x.dim() + dim
    output = distfunc.all_gather_tensor(
        x.transpose(0, dim).contiguous(), 0, list(range(world_size)))
    output = output.transpose(0, dim)
    return output
