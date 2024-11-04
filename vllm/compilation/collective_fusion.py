import operator
from typing import List, Tuple

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import (Match, PatternMatcherPass,
                                             fwd_only, register_replacement)

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.compilation.inductor_pass import InductorPass
from vllm.compilation.utils import (find_auto_fn, find_fn, find_getitem,
                                    last_node_in_match)
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.logger import init_logger
from vllm.utils import direct_register_custom_op

logger = init_logger(__name__)

use_flux = False
if envs.VLLM_USE_FLUX:
    try:
        import flux
        use_flux = True
        logger.info("USING FLUX")
    except ImportError:
        use_flux = False

# TODO: factor out somehow
TP_GROUP_NAME = "tp:0"


# how to do this properly?
def get_world_name() -> str:
    return torch.distributed.group.WORLD.group_name


# This check is a hack
def should_slice(shape) -> bool:
    n_slices = get_tensor_model_parallel_world_size()
    return (shape[0] % n_slices == 0 and shape[0] >= 128)


def slice_residual(residual) -> List[torch.Tensor]:
    n_slices = get_tensor_model_parallel_world_size()
    return torch.chunk(residual, n_slices, dim=0)


def match_gemm_rs_ag_gemm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    gemm_2_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = torch.ops.higher_order.auto_functionalized(
        torch.ops.vllm.inplace_all_reduce.default,
        tensor=mm_1,
        group_name=TP_GROUP_NAME)
    all_reduce = all_reduce[1]

    norm_res = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=residual,
        weight=rms_norm_weight,
        epsilon=1e-05)
    normalized = norm_res[1]
    new_residual = norm_res[2]

    gemm_2_w_perm = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
    mm_2 = torch.ops.aten.mm.default(normalized, gemm_2_w_perm)

    return mm_2, new_residual


def gemm_rs_ag_gemm_fake(
    residual: torch.Tensor,
    my_residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    gemm_2_weights: torch.Tensor,
    first_layer: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if should_slice(gemm_1_activations.shape) and first_layer:
        res_slices = slice_residual(residual)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        my_residual = split_1[0]
    else:
        my_residual = residual

    # verify the type is always correct
    mm_res = torch.empty(
        (gemm_1_activations.shape[0], gemm_2_weights.shape[0]),
        device=gemm_1_activations.device,
        dtype=gemm_1_activations.dtype)

    return (mm_res, my_residual, residual)


# TODO: factor out groupnames, etc.
def get_gemm_rs_ag_gemm(use_flux: bool,
                        gemm_1_type,
                        gemm_1_weights: torch.Size,
                        gemm_2_type,
                        gemm_2_weights: torch.Size):

    if use_flux:
        gemm_rs_op = flux.GemmRS(
            get_tp_group().device_group,
            1,  # One node
            8192,  # Max M. TODO: Pass in correctly.
            gemm_1_weights[0],  # N
            # TODO: Pass in input dtype correctly.
            # TODO: It would be nicer to modify flux to dispatch based on dtype
            # at run time, but I don't know what the downside would be.
            # Similar comment for max m.
            gemm_1_type,
            # Note: transpose_weight=False means that B is transposed
            transpose_weight=False,
            # Note: bfloat16 requires fuse_reduction=False.
            fuse_reduction=False,
        )

        ag_gemm_op = flux.AGKernel(
            get_tp_group().device_group,
            1,  # One node
            8192,  # Max M. TODO: Pass in correctly.
            gemm_2_weights[0],  # N
            gemm_2_weights[1],  # K
            # TODO: Pass in input dtype correctly.
            # TODO: It would be nicer to modify flux to dispatch based on dtype
            # at run time, but I don't know what the downside would be.
            # Similar comment for max m.
            gemm_2_type,
            gemm_2_type,
            # Note: transpose_weight=False means that B is transposed
            transpose_weight=False,
            # Note: if local_copy=True, I hit the following runtime error:
            # /flux/src/all_gather/ths_op/all_gather_gemm_kernel.cc:648
            #   Check failed: 33554432((input.numel() * input.element_size()))
            #                 == 139836453421056((this->chunk_size))
            local_copy=False,
        )

        gemm_rs = lambda act, wt: gemm_rs_op.forward(act, wt).squeeze(0)
        ag_gemm = lambda act, wt: ag_gemm_op.forward(act, wt)

        gemm_1_str = str(gemm_1_type).removeprefix("torch.")
        gemm_2_str = str(gemm_2_type).removeprefix("torch.")
        name = (f"gemm_rs_ag_gemm_{gemm_1_str}_{gemm_1_weights[0]}_"
                f"{gemm_2_str}_{gemm_2_weights[0]}_{gemm_2_weights[1]}")
    else:
        group_name = get_world_name()

        gemm_rs = lambda act, wt: \
            torch.ops.symm_mem.fused_matmul_reduce_scatter.default(
                act, wt.transpose(1, 0), 'avg', 0, group_name)

        ag_gemm = lambda act, wt: \
            torch.ops.symm_mem.fused_all_gather_matmul.default(
                act, [wt.transpose(1, 0)], 0, group_name)[1]

        name = "gemm_rs_ag_gemm"

    def gemm_rs_ag_gemm(
            residual: torch.Tensor, old_my_residual: torch.Tensor,
            gemm_1_weights: torch.Tensor, gemm_1_activations: torch.Tensor,
            rms_norm_weight: torch.Tensor, gemm_2_weights: torch.Tensor,
            first_layer: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if first_layer and should_slice(residual.shape):
            res_slices = slice_residual(residual)
            slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
            residual_chunk = torch.ops.aten.split.Tensor(residual, slice_size)
            my_residual = residual_chunk[0]
        else:
            my_residual = residual #.clone()
            slice_size = residual.shape[0]

        if not should_slice(residual.shape):
            output = torch.matmul(gemm_1_activations,
                                  gemm_1_weights.transpose(1, 0))
            reduced_output = tensor_model_parallel_all_reduce(output)

            ops.fused_add_rms_norm(input=reduced_output,
                                   residual=my_residual,
                                   weight=rms_norm_weight,
                                   epsilon=1e-05)

            mm_2 = torch.matmul(reduced_output, gemm_2_weights.transpose(1, 0))
            return mm_2, my_residual, my_residual.clone()
        else:
            output = gemm_rs(gemm_1_activations, gemm_1_weights)

            ops.fused_add_rms_norm(input=output,
                                   residual=my_residual,
                                   weight=rms_norm_weight,
                                   epsilon=1e-05)

            residual_1 = residual if first_layer else old_my_residual
            slice_scatter = torch.ops.aten.slice_scatter.default(
                residual_1, my_residual, 0, 0, slice_size)
            split_2 = torch.ops.aten.split.Tensor(slice_scatter, slice_size)

            # TODO: can we avoid clone here?
            new_residual = split_2[0]  #.clone()

            mm_2 = ag_gemm(output, gemm_2_weights)

            return mm_2[0], new_residual, slice_scatter

    if not hasattr(torch.ops.vllm, name):
        logger.info("registering torch.ops.vllm.%s", name)
        direct_register_custom_op(
            name,
            gemm_rs_ag_gemm,
            mutates_args=[],
            fake_impl=gemm_rs_ag_gemm_fake
        )
        assert getattr(torch.ops.vllm, name)

    return getattr(torch.ops.vllm, name).default


def match_final(my_residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                gemm_1_activations: torch.Tensor,
                rms_norm_weights: torch.Tensor) -> torch.Tensor:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = torch.ops.higher_order.auto_functionalized(
        torch.ops.vllm.inplace_all_reduce.default,
        tensor=mm_1,
        group_name=TP_GROUP_NAME)
    all_reduce = all_reduce[1]

    norm_res = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=my_residual,
        weight=rms_norm_weights,
        epsilon=1e-05)
    normalized = norm_res[1]

    return normalized


# Register this as a custom op since all reduce cannot be torch.compiled.
def replace_final(my_residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                  gemm_1_activations: torch.Tensor,
                  rms_norm_weights: torch.Tensor) -> torch.Tensor:
    permute_254 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, permute_254)

    reduced = tensor_model_parallel_all_reduce(mm_1)

    if should_slice(gemm_1_activations.shape):
        if True: #not use_flux:
            group_name = get_world_name()
            world_size = get_tensor_model_parallel_world_size()
            all_gather = (
                torch.ops._c10d_functional.all_gather_into_tensor.default(
                    my_residual, world_size, group_name))
            wait_tensor = torch.ops._c10d_functional.wait_tensor.default(
                all_gather)
        else:
            wait_tensor = tensor_model_parallel_all_gather(my_residual)
    else:
        wait_tensor = my_residual

    ops.fused_add_rms_norm(
        input=reduced,
        residual=wait_tensor,
        weight=rms_norm_weights,
        epsilon=1e-05)

    return reduced


def replace_final_fake(my_residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                       gemm_1_activations: torch.Tensor,
                       rms_norm_weights: torch.Tensor) -> torch.Tensor:
    return torch.empty([gemm_1_activations.shape[0], my_residual.shape[1]],
                       dtype=my_residual.dtype,
                       device=my_residual.device)


direct_register_custom_op(
    "gemm_ag_final",
    replace_final,
    mutates_args=[],
    fake_impl=replace_final_fake
)


class CollectiveFusionPass(InductorPass):

    def __init__(self):
        self.gemm_rs_ag_gemm_pattern = PatternMatcherPass()
        self.final_pattern = PatternMatcherPass()
        self.matches: List[Match] = []

        x = torch.empty([4, 4], device='cuda')
        w = torch.empty([4, 4], device='cuda')
        resid = torch.empty([4, 4], device='cuda')
        resid_w = torch.empty([4, 4], device='cuda')
        x2 = torch.empty([4, 4], device='cuda')
        inputs = [resid, x, w, resid_w, x2]

        register_replacement(match_gemm_rs_ag_gemm,
                             match_gemm_rs_ag_gemm,
                             inputs,
                             fwd_only, [self.gemm_rs_ag_gemm_pattern],
                             extra_check=lambda m: self.record_match(m))

        final_inputs = [x, w, resid, resid_w]
        register_replacement(
            match_final,
            torch.ops.vllm.gemm_ag_final,
            #replace_final,
            final_inputs,
            fwd_only,
            [self.final_pattern])

    def record_match(self, match: Match) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        self.matches.append(match)

        # Return False to prevent automatic replacement.
        return False

    def process_matches(self, graph: fx.Graph):
        nodes = list(graph.nodes)

        def find_min_index(match: Match) -> int:
            return min(match.nodes, key=lambda x: nodes.index(x))

        # "sort" matches in topo order.
        matches = sorted(self.matches, key=lambda x: find_min_index(x))

        res_replacements: List[fx.Node] = []
        my_res_replacements: List[fx.Node] = []

        for match in matches:
            last_node = last_node_in_match(match)

            with graph.inserting_after(last_node):
                kwargs = match.kwargs
                kwargs["first_layer"] = match == matches[0]
                kwargs["residual"] = res_replacements[-1] if len(
                    res_replacements) > 0 else match.kwargs["residual"]
                kwargs["old_my_residual"] = my_res_replacements[-1] if len(
                    my_res_replacements) > 0 else match.kwargs["residual"]

                # TODO: use get
                gemm_1 = kwargs["gemm_1_weights"].meta["val"]
                gemm_2 = kwargs["gemm_2_weights"].meta["val"]

                fused_node = graph.call_function(get_gemm_rs_ag_gemm(
                    use_flux, gemm_1.dtype, gemm_1.shape, gemm_2.dtype, gemm_2.shape),
                                                 kwargs=kwargs)

                graph.inserting_after(fused_node)
                result_node_new = graph.call_function(operator.getitem,
                                                      (fused_node, 0))
                residual_node_new = graph.call_function(
                    operator.getitem, (fused_node, 1))
                my_residual_node_new = graph.call_function(
                    operator.getitem, (fused_node, 2))
                res_replacements.append(residual_node_new)
                my_res_replacements.append(my_residual_node_new)

            rms_node = find_auto_fn(reversed(match.nodes),
                                    torch.ops._C.fused_add_rms_norm.default)
            gemm_node = find_fn(reversed(match.nodes),
                                torch.ops.aten.mm.default)
            assert rms_node is not None
            assert gemm_node is not None

            assert len(rms_node.users) == 2
            assert len(gemm_node.users) == 1 or len(gemm_node.users) == 2

            residual_getter_node = find_getitem(rms_node, 2)
            assert residual_getter_node is not None
            residual_getter_node.replace_all_uses_with(residual_node_new)
            gemm_node.replace_all_uses_with(result_node_new)

        # Finally, remove matched nodes
        graph.eliminate_dead_code()
        assert all(node not in graph.nodes for match in matches for node in match.nodes)

    def __call__(self, graph: fx.Graph):
        self.dump_graph(graph, "before_collective_fusion")
        count = self.gemm_rs_ag_gemm_pattern.apply(graph)
        logger.info("fused gemm match count = %d", len(self.matches))

        # Don't apply final pattern unless we've matched and replaced the
        # gemm+collective ops.
        if len(self.matches) > 0:
            count = self.final_pattern.apply(graph)
            logger.info("final match count = %d", count)
            self.process_matches(graph)

        self.dump_graph(graph, "after_collective_fusion")
        self.matches.clear()
