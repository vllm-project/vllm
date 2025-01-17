import contextlib
import operator
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import (Match, PatternMatcherPass,
                                             fwd_only, register_replacement)

import vllm.envs as envs
from vllm.compilation.fx_utils import (find_auto_fn, find_fn, find_getitem,
                                       last_node_in_match)
from vllm.config import CompilationConfig
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (
    get_group_from_group_name, get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.utils import direct_register_custom_op

from .inductor_pass import get_pass_context
from .utils import use_cc_kernels
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

use_flux = False
if envs.VLLM_USE_FLUX:
    try:
        import flux
        use_flux = True
        logger.info("Using flux kernels for collective communication fusion.")
    except ImportError:
        logger.info("Attempting to use flux but flux not installed.")
        use_flux = False


def get_world_name() -> str:
    return torch.distributed.group.WORLD.group_name


def residual_slice_shape(residual: torch.Tensor, rank: int) -> int:
    n_slices = get_tensor_model_parallel_world_size()
    assert residual.size(0) % n_slices == 0
    return residual.size(0) // n_slices


def match_gemm_rs_ag_gemm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
    gemm_2_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    norm_res = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-05)
    normalized = norm_res[1]
    new_residual = norm_res[2]

    gemm_2_w_perm = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
    mm_2 = torch.ops.aten.mm.default(normalized, gemm_2_w_perm)

    return mm_2, new_residual


def get_gemm_rs_ag_gemm(max_m: int, gemm_1_type: torch.dtype,
                        gemm_1_weights: torch.Size, gemm_2_type: torch.dtype,
                        gemm_2_weights: torch.Size, tp_group_name: str,
                        is_static_shape: bool) -> Tuple[Callable, Callable]:

    group = get_group_from_group_name(tp_group_name)
    device_group = group.device_group
    rank = group.rank_in_group

    if use_flux:
        gemm_1_str = str(gemm_1_type).removeprefix("torch.")
        gemm_2_str = str(gemm_2_type).removeprefix("torch.")
        group_str = tp_group_name.replace(":", "_")
        name = (f"gemm_rs_ag_gemm_{max_m}_{gemm_1_str}_{gemm_1_weights[0]}_"
                f"{gemm_2_str}_{gemm_2_weights[0]}_{gemm_2_weights[1]}_"
                f"{group_str}")

        if not hasattr(torch.ops.vllm, name):
            gemm_rs_op = flux.GemmRS(
                device_group,
                1,  # One node
                max_m,  # max M
                gemm_1_weights[0],  # N
                # TODO: It would be nicer to modify flux to dispatch based on
                # dtype at run time, but I don't know what the downside would
                # be. Similar comment for max m.
                gemm_1_type,
                # Note: transpose_weight=False means that B is transposed
                transpose_weight=False,
                # Note: bfloat16 requires fuse_reduction=False.
                fuse_reduction=False,
            )

            ag_gemm_op = flux.AGKernel(
                device_group,
                1,  # One node
                max_m,  # max M
                gemm_2_weights[0],  # N
                gemm_2_weights[1],  # K
                # TODO: It would be nicer to modify flux to dispatch based on
                # dtype at run time, but I don't know what the downside would
                # be. Similar comment for max m.
                gemm_2_type,
                gemm_2_type,
                # Note: transpose_weight=False means that B is transposed
                transpose_weight=False,
                # Note: if local_copy=True, I hit the following runtime error:
                # /flux/src/all_gather/ths_op/all_gather_gemm_kernel.cc:648
                # Check failed: 33554432((input.numel() * input.element_size()))
                #                 == 139836453421056((this->chunk_size))
                local_copy=False,
            )

            def gemm_rs(act, wt):
                return gemm_rs_op.forward(act, wt).squeeze(0)

            def ag_gemm(act, wt):
                return ag_gemm_op.forward(act, wt)

    else:
        group_str = tp_group_name.replace(":", "_")
        name = f"gemm_rs_ag_gemm_{group_str}"

        if not hasattr(torch.ops.vllm, name):
            world_group_name = get_world_name()

            def gemm_rs(act, wt):
                return torch.ops.symm_mem.fused_matmul_reduce_scatter.default(
                    act, wt.transpose(1, 0), 'avg', 0, world_group_name)

            def ag_gemm(act, wt):
                _, out = torch.ops.symm_mem.fused_all_gather_matmul.default(
                    act, [wt.transpose(1, 0)], 0, world_group_name)
                return out[0]

    def gemm_rs_ag_gemm(
        residual: torch.Tensor,
        old_my_residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weights: torch.Tensor,
        gemm_2_weights: torch.Tensor,
        first_layer: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        do_split = use_cc_kernels(residual.size(0))

        if first_layer and do_split:
            slice_shape = residual_slice_shape(residual, rank)
            residual_chunk = torch.ops.aten.split.Tensor(residual, slice_shape)
            my_residual = residual_chunk[0]
        else:
            my_residual = residual
            slice_shape = residual.size(0)

        if not do_split:
            output = torch.ops.aten.mm.default(gemm_1_activations,
                                               gemm_1_weights.transpose(1, 0))
            reduced_output = tensor_model_parallel_all_reduce(output)

            torch.ops._C.fused_add_rms_norm.default(input=reduced_output,
                                                    residual=my_residual,
                                                    weight=rms_norm_weights,
                                                    epsilon=1e-05)

            mm_2 = torch.ops.aten.mm.default(reduced_output,
                                             gemm_2_weights.transpose(1, 0))

            return mm_2, my_residual, my_residual.clone()
        else:
            output = gemm_rs(gemm_1_activations, gemm_1_weights)

            torch.ops._C.fused_add_rms_norm.default(input=output,
                                                    residual=my_residual,
                                                    weight=rms_norm_weights,
                                                    epsilon=1e-05)

            residual_1 = residual if first_layer else old_my_residual
            slice_scatter = torch.ops.aten.slice_scatter.default(
                residual_1, my_residual, 0, 0, slice_shape)
            split_2 = torch.ops.aten.split.Tensor(slice_scatter, slice_shape)
            new_residual = split_2[0]

            mm_2 = ag_gemm(output, gemm_2_weights)

            return mm_2, new_residual, slice_scatter

    def gemm_rs_ag_gemm_static(
        residual: torch.Tensor,
        old_my_residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weights: torch.Tensor,
        gemm_2_weights: torch.Tensor,
        first_layer: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first_layer:
            slice_shape = residual_slice_shape(residual, rank)
            residual_chunk = torch.ops.aten.split.Tensor(residual, slice_shape)
            my_residual = residual_chunk[0]
        else:
            slice_shape = residual.size(0)
            my_residual = residual

        output = gemm_rs(gemm_1_activations, gemm_1_weights)

        torch.ops._C.fused_add_rms_norm.default(input=output,
                                                residual=my_residual,
                                                weight=rms_norm_weights,
                                                epsilon=1e-05)

        residual_1 = residual if first_layer else old_my_residual
        slice_scatter = torch.ops.aten.slice_scatter.default(
            residual_1, my_residual, 0, 0, slice_shape)
        split_2 = torch.ops.aten.split.Tensor(slice_scatter, slice_shape)
        new_residual = split_2[0]

        mm_2 = ag_gemm(output, gemm_2_weights)

        return mm_2, new_residual, slice_scatter

    def gemm_rs_ag_gemm_fake(
        residual: torch.Tensor,
        my_residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weights: torch.Tensor,
        gemm_2_weights: torch.Tensor,
        first_layer: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first_layer and use_cc_kernels(residual.size(0)):
            slice_shape = residual_slice_shape(residual, rank)
            my_residual = torch.empty((slice_shape, residual.size(1)),
                                      device=residual.device,
                                      dtype=residual.dtype)
        else:
            my_residual = residual

        # TODO: verify the type is always correct
        mm_res = torch.empty(
            (gemm_1_activations.size(0), gemm_2_weights.size(0)),
            device=gemm_1_activations.device,
            dtype=gemm_1_activations.dtype)

        return (mm_res, my_residual, residual)

    if not hasattr(torch.ops.vllm, name):
        logger.info("registering torch.ops.vllm.%s", name)
        grag = gemm_rs_ag_gemm_static if is_static_shape else gemm_rs_ag_gemm
        direct_register_custom_op(name,
                                  grag,
                                  mutates_args=[],
                                  fake_impl=gemm_rs_ag_gemm_fake)
        assert getattr(torch.ops.vllm, name)

    return getattr(torch.ops.vllm, name).default, gemm_rs_ag_gemm_fake


def match_final(
    my_residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
) -> torch.Tensor:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    norm_res = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=my_residual,
        weight=rms_norm_weights,
        epsilon=1e-05)
    normalized = norm_res[1]

    return normalized


# Register this as a custom op since all gather cannot be torch.compiled yet.
def gemm_ag_final(my_residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                  gemm_1_activations: torch.Tensor,
                  rms_norm_weights: torch.Tensor) -> torch.Tensor:
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations,
                                     gemm_1_weights.transpose(1, 0))

    reduced = tensor_model_parallel_all_reduce(mm_1)

    if use_cc_kernels(reduced.size(0)):
        wait_tensor = tensor_model_parallel_all_gather(my_residual)
    else:
        assert reduced.size() == my_residual.size()
        wait_tensor = my_residual

    torch.ops._C.fused_add_rms_norm.default(input=reduced,
                                            residual=wait_tensor,
                                            weight=rms_norm_weights,
                                            epsilon=1e-05)

    return reduced


def gemm_ag_final_static(my_residual: torch.Tensor,
                         gemm_1_weights: torch.Tensor,
                         gemm_1_activations: torch.Tensor,
                         rms_norm_weights: torch.Tensor) -> torch.Tensor:
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations,
                                     gemm_1_weights.transpose(1, 0))

    reduced = tensor_model_parallel_all_reduce(mm_1)

    wait_tensor = tensor_model_parallel_all_gather(my_residual)

    torch.ops._C.fused_add_rms_norm.default(input=reduced,
                                            residual=wait_tensor,
                                            weight=rms_norm_weights,
                                            epsilon=1e-05)

    return reduced


def gemm_ag_final_fake(my_residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                       gemm_1_activations: torch.Tensor,
                       rms_norm_weights: torch.Tensor) -> torch.Tensor:
    return torch.empty([gemm_1_activations.size(0),
                        my_residual.size(1)],
                       dtype=my_residual.dtype,
                       device=my_residual.device)


direct_register_custom_op("gemm_ag_final",
                          gemm_ag_final,
                          mutates_args=[],
                          fake_impl=gemm_ag_final_fake)

#direct_register_custom_op("gemm_ag_final_static",
#                          gemm_ag_final_static,
#                          mutates_args=[],
#                          fake_impl=gemm_ag_final_fake)


def trace_fn(fn: Any, args: Sequence[Any]) -> fx.GraphModule:
    from torch._inductor.virtualized import NullHandler, V
    context = (V.fake_mode if
               (not isinstance(V.fake_mode, NullHandler) or
                (V.fake_mode is None)) else contextlib.nullcontext())

    with context:
        return fwd_only(fn, args)


class CollectiveFusionPass(VllmInductorPass):

    _instance: 'Optional[CollectiveFusionPass]' = None

    @classmethod
    def instance(cls, config: CompilationConfig) -> "CollectiveFusionPass":
        """
        Get the singleton instance of the CollectiveFusionPass.
        If the instance exists, the config is updated but
        initialization is not repeated.
        """
        if cls._instance is None:
            cls._instance = CollectiveFusionPass(config)
        else:
            cls._instance.config = config
        return cls._instance

    def __init__(self, config: CompilationConfig):
        assert self.__class__._instance is None, \
            "CollectiveFusionPass singleton instance already exists"
        super().__init__(config)

        self.gemm_rs_ag_gemm_pattern = PatternMatcherPass()
        self.final_pattern = PatternMatcherPass()
        self.matches: List[Match] = []

        x = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        w = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        resid = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        resid_w = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        x2 = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        inputs = [resid, x, w, resid_w, x2]
        final_inputs = [x, w, resid, resid_w]

        register_replacement(match_gemm_rs_ag_gemm,
                             match_gemm_rs_ag_gemm,
                             inputs,
                             fwd_only, [self.gemm_rs_ag_gemm_pattern],
                             extra_check=lambda m: self.record_match(m))

        # Nice TODO: handle static shape
        register_replacement(match_final, torch.ops.vllm.gemm_ag_final,
                             final_inputs, fwd_only, [self.final_pattern])

    def is_static_shape(self):
        pass_context = get_pass_context()
        return pass_context.runtime_shape is not None

    # TODO: Add type check
    def should_rewrite(self, match: Match) -> bool:
        pass_context = get_pass_context()
        if pass_context.runtime_shape is None:
            return self.config.enable_dynamic_collective_fusion
        return use_cc_kernels(pass_context.runtime_shape)

    def record_match(self, match: Match) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        if self.should_rewrite(match):
            self.matches.append(match)

        # Return False to prevent automatic replacement.
        return False

    def collect_args(self, kwargs) -> Tuple[List[Any], List[Any]]:
        arg_names = {
            "residual": 0,
            "old_my_residual": 1,
            "gemm_1_weights": 2,
            "gemm_1_activations": 3,
            "rms_norm_weights": 4,
            "gemm_2_weights": 5,
            "first_layer": 6,
        }
        args = [None] * len(arg_names)
        node_args = [None] * len(arg_names)
        for k, v in kwargs.items():
            idx = arg_names[k]
            if isinstance(v, torch.fx.Node):
                if v.meta.get("val") is not None:
                    args[idx] = v.meta["val"]
            else:
                args[idx] = v
            node_args[idx] = v
        return node_args, args

    def process_matches(self, graph: fx.Graph) -> None:

        def find_min_index(match: Match) -> int:
            return min(match.nodes, key=lambda x: nodes.index(x))

        nodes = list(graph.nodes)

        # "sort" matches in topo order.
        matches = sorted(self.matches, key=lambda x: find_min_index(x))

        res_replacements: List[fx.Node] = []
        my_res_replacements: List[fx.Node] = []

        max_m = self.config.max_num_batched_tokens
        logger.info("max m = %d", max_m)

        n = 0
        for match in matches:
            last_node = last_node_in_match(match)
            first_layer = match == matches[0]
            n = n + 1

            with graph.inserting_after(last_node):
                kwargs = match.kwargs
                kwargs["first_layer"] = first_layer
                kwargs["residual"] = res_replacements[-1] if len(
                    res_replacements) > 0 else match.kwargs["residual"]
                kwargs["old_my_residual"] = my_res_replacements[-1] if len(
                    my_res_replacements) > 0 else match.kwargs["residual"]

                gemm_1 = kwargs["gemm_1_weights"].meta.get("val")
                gemm_2 = kwargs["gemm_2_weights"].meta.get("val")
                if gemm_1 is None or gemm_2 is None:
                    raise ValueError("Missing 'val' in gemm weights meta data")

                # Extract group_name from matched code.  Use to
                # generate proper replacement code.
                ar_node = find_fn(match.nodes,
                                  torch.ops.vllm.all_reduce.default)
                assert ar_node is not None
                tp_group_name = ar_node.args[1]

                fused_gemm_func, fused_gemm_fake_func = get_gemm_rs_ag_gemm(
                    max_m, gemm_1.dtype, gemm_1.shape, gemm_2.dtype,
                    gemm_2.shape, tp_group_name, self.is_static_shape())

                fused_node = graph.call_function(fused_gemm_func,
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

            # Update meta data by using the fake function (optional)
            if False:
                node_args, args = self.collect_args(kwargs)
                fake_mod = trace_fn(fused_gemm_fake_func, args)
                outputs = [n for n in fake_mod.graph.nodes if n.op == 'output']
                assert len(outputs) == 1
                metas = []
                for out in outputs[0].args[0]:
                    metas.append(out.meta["val"])
                tm = tuple(metas)

                fused_node.meta["val"] = tm
                result_node_new.meta["val"] = tm[0]
                residual_node_new.meta["val"] = tm[1]
                my_residual_node_new.meta["val"] = tm[2]

            residual_getter_node = find_getitem(rms_node, 2)
            assert residual_getter_node is not None
            residual_getter_node.replace_all_uses_with(residual_node_new)
            gemm_node.replace_all_uses_with(result_node_new)

        # Finally, remove matched nodes
        graph.eliminate_dead_code()
        assert all(node not in graph.nodes for match in matches
                   for node in match.nodes)

    def __call__(self, graph: fx.Graph):
        pass_context = get_pass_context()

        if (pass_context.runtime_shape is None
                and not self.config.enable_dynamic_collective_fusion):
            logger.info("CollectiveFusionPass disabled for general shape.")
            return
        else:
            logger.info("CollectiveFusionPass shape=%s",
                        pass_context.runtime_shape)

        #
        # TODO: would be nice to disable/assert if chunk prefill size is too
        # small but that info is not easily available here.
        #

        self.dump_graph(graph, "before_collective_fusion")
        self.gemm_rs_ag_gemm_pattern.apply(graph)
        logger.info("fused gemm match count = %d", len(self.matches))

        # Don't apply final pattern unless we've matched and replaced the
        # gemm+collective ops.
        if len(self.matches) > 0:
            count = self.final_pattern.apply(graph)
            logger.info("final match count = %d", count)
            self.process_matches(graph)

        self.dump_graph(graph, "after_collective_fusion")
        self.matches.clear()
