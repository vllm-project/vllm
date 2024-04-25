import collections
from typing import DefaultDict, TypeVar, Generic

import torch
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
import torch._inductor.scheduler as sched
import torch._inductor.dependencies as dependencies
from torch._inductor.virtualized import V

vllm_lib = torch.library.Library("vllm", "DEF")


# Fixed in PT 2.2
def nodeuser_hash(self):
    return hash((self.node.get_name(), self.can_inplace, self.is_weak))


def nodeuser_eq(self, other):
    return (self.get_name() == other.get_name()
            and self.can_inplace == other.can_inplace
            and self.is_weak == other.is_weak)


sched.NodeUser.__hash__ = nodeuser_hash
sched.NodeUser.__eq__ = nodeuser_eq


def pt22_compute_dependencies(self):
    """
    Create dependency edges between nodes, handling aliasing and
    mutation properly.
    """
    T = TypeVar("T")

    class DedupList(Generic[T]):
        """
        This data structure behaves like a list except it makes sure the
        elements remain unique.
        Normally one could use a set/dict for this purpose however
        the list in question gets elements appended as it is being
        iterated over which means that we need to keep the list
        semantics.
        """

        def __init__(self, items=None, membership=None):
            self.items = items or list()
            self.membership = membership or set()

        def append(self, node_user: T) -> None:
            if node_user in self.membership:
                return
            self.items.append(node_user)
            self.membership.add(node_user)

        def __add__(self, other: "DedupList[T]") -> "DedupList[T]":
            new_membership = set.union(self.membership, other.membership)
            new_items = self.items + [
                x for x in other.items if x not in self.membership
            ]
            return DedupList(new_items, new_membership)

    name_to_users: DefaultDict[
        str, DedupList[sched.NodeUser]] = collections.defaultdict(DedupList)

    # handle aliasing by using python aliasing in name_to_users
    # if foo aliases bar then we will make name_to_users["foo"] point
    # to the same python list as name_to_users["bar"]
    for node1 in self.nodes:
        node1_name = node1.get_name()
        for node2_name in node1.get_aliases():
            if node1_name in name_to_users and node2_name in name_to_users:
                # merge the two
                list1 = name_to_users[node1_name]
                list2 = name_to_users[node2_name]
                combined = list1 + list2
                for key in name_to_users:
                    if name_to_users[key] is list1 or name_to_users[
                            key] is list2:
                        name_to_users[key] = combined
            elif node1_name in name_to_users:
                name_to_users[node2_name] = name_to_users[node1_name]
            else:
                name_to_users[node1_name] = name_to_users[node2_name]

    def rename(n):
        if n in self.mutation_renames:
            return rename(self.mutation_renames[n])
        return n

    def dep_closure(node_name):
        reachable_names = {node_name}
        node = self.name_to_node[node_name]
        write_dep = list(node.read_writes.writes)[0]
        for read_dep in node.read_writes.reads:
            if (read_dep.name in self.name_to_node
                    and isinstance(read_dep, dependencies.MemoryDep)
                    and isinstance(write_dep, dependencies.MemoryDep)
                    and read_dep.index == write_dep.index
                    and read_dep.size == write_dep.size):
                reachable_names.update(dep_closure(read_dep.name))
        return reachable_names

    def add_user(used_by_name, user_node, can_inplace=False, is_weak=False):
        name_to_users[rename(used_by_name)].append(
            sched.NodeUser(user_node, can_inplace, is_weak))

    for node in self.nodes:
        # a node will mutate either 0 or 1 buffers
        for alt_name in node.get_mutations():
            alt_name = rename(alt_name)
            # this node must run after the prior writer
            add_user(alt_name, node)
            node.add_mutation_dep(dependencies.StarDep(alt_name))
            for other_node in name_to_users[alt_name].items:
                # this node must run after all prior readers
                other_name = rename(other_node.get_name())
                known_dep_node_names = dep_closure(node.get_name())
                if other_name not in known_dep_node_names:
                    # If this node already directly or indirectly depends on other_node,
                    # we don't need to insert an extra dep.
                    node.add_mutation_dep(dependencies.WeakDep(other_name))
                    add_user(other_name, node, is_weak=True)

        # add normal non-mutation dependencies
        for read in node.read_writes.reads:
            is_weak = isinstance(read, dependencies.WeakDep)
            add_user(read.name, node, node.can_inplace(read), is_weak)

        node.update_mutated_names(self.mutation_renames)

        # update our renaming scheme for the next iteration
        for alt_name in node.get_mutations():
            self.mutation_renames[rename(alt_name)] = node.get_name()
            self.mutation_renames[alt_name] = node.get_name()
            self.mutation_real_name[
                node.get_name()] = self.mutation_real_name.get(
                    alt_name, alt_name)

    # make sure outputs aren't dead-code-eliminated
    for node_name in V.graph.get_output_names():
        add_user(node_name, sched.OutputNode(dependencies.StarDep(node_name)))

    # make sure input mutation isn't dead-code-eliminated
    for name in self.mutation_renames:
        if name in V.graph.graph_inputs:
            add_user(name, sched.OutputNode(dependencies.StarDep(name)))
            V.graph.mutated_inputs.add(name)

    inp_names = {
        name: index
        for index, name in enumerate(V.graph.graph_inputs.keys())
    }
    V.graph.mutated_input_idxs = [
        inp_names[name] for name in V.graph.mutated_inputs
    ]

    # copy users information onto the nodes
    for node in self.nodes:
        node.set_users(name_to_users[node.get_name()].items)

    # populate inverse_users
    for node in self.nodes:
        for user in node.users:
            user.node.inverse_users.append(node)


sched.Scheduler.compute_dependencies = pt22_compute_dependencies


# Available starting PT 2.2
class NoneLayout(ir.IRNode):

    def __init__(self, device):
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        return 0

    def as_fixed(self):
        return self


# Available starting PT 2.2
class MutationOutput(ir.ExternKernel):

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def __init__(self, layout, input, parent):
        super().__init__(None, layout, [input, parent], ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False

    def is_no_op(self):
        return True

    def has_side_effects(self):
        return True

    def get_alias_names(self):
        return [self.inputs[0].get_name()]


class VllmCudaKernel(ir.FallbackKernel):

    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    @classmethod
    def create(cls, kernel, *args, mutated_inputs=None, **kwargs) -> None:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        packed = cls(
            NoneLayout(tensor_args[0].get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            schema=schema,
        )

        # Mark inplace inputs as mutated
        def mark_mutation(x):
            if isinstance(x.data, ir.BaseView):
                x = x.data.unwrap_view()
            MutationOutput(x.layout, x, packed)

        for kernel_input_idx in mutated_inputs:
            kernel_input = args[kernel_input_idx]
            # V.graph.mark_buffer_mutated(kernel_input.get_name())
            mark_mutation(kernel_input)
            # MutationOutput(kernel_input.layout, kernel_input, packed)


def register_vllm_lowering(op, mutating_inputs):
    lowering.fallbacks.add(op)

    @lowering.register_lowering(op, type_promotion_kind=None)
    def op_lowering(
        *args,
        **kwargs,
    ):
        VllmCudaKernel.create(
            op.default,
            *args,
            **kwargs,
            mutated_inputs=mutating_inputs,
        )
        returns = [args[i] for i in mutating_inputs]
        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)
