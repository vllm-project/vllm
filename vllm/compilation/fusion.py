import operator

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (Match, PatternMatcherPass,
                                             fwd_only, register_replacement)

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def rms_pattern_static(result: torch.Tensor, result_rms: torch.Tensor,
                       input: torch.Tensor, weight: torch.Tensor,
                       scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops._C.rms_norm.default,
                              result=result_rms,
                              input=input,
                              weight=weight,
                              epsilon=1e-5)
    at2 = auto_functionalized(torch.ops._C.static_scaled_fp8_quant.default,
                              result=result,
                              input=at1[1],
                              scale=scale)

    # result
    return at2[1]


def rms_replacement_static(result: torch.Tensor, result_rms: torch.Tensor,
                           input: torch.Tensor, weight: torch.Tensor,
                           scale: torch.Tensor):
    at = auto_functionalized(torch.ops._C.rms_norm_static_fp8_quant.default,
                             result=result,
                             input=input,
                             weight=weight,
                             scale=scale,
                             epsilon=1e-5)

    # result
    return at[1]


def rms_pattern_residual_static(result: torch.Tensor, input: torch.Tensor,
                                residual: torch.Tensor, weight: torch.Tensor,
                                scale: torch.Tensor):
    at = auto_functionalized(torch.ops._C.fused_add_rms_norm.default,
                             input=input,
                             residual=residual,
                             weight=weight,
                             epsilon=1e-5)
    at1 = auto_functionalized(torch.ops._C.static_scaled_fp8_quant.default,
                              result=result,
                              input=at[1],
                              scale=scale)

    # result, residual
    return at1[1], at[2]


def rms_replacement_residual_static(result: torch.Tensor, input: torch.Tensor,
                                    residual: torch.Tensor,
                                    weight: torch.Tensor, scale: torch.Tensor):
    at = auto_functionalized(
        torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,
        result=result,
        input=input,
        residual=residual,
        weight=weight,
        scale=scale,
        epsilon=1e-5)
    # result, residual
    return at[1], at[2]


my_patterns = PatternMatcherPass()


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp8(*args, **kwargs):
    return torch.empty(*args,
                       **kwargs,
                       dtype=torch.float8_e4m3fn,
                       device="cuda")


def get_patterns():
    my_patterns = PatternMatcherPass(pass_name="fusion_pass")

    inputs = [
        empty_fp8(5, 4),
        empty_bf16(5, 4),
        empty_bf16(5, 4),
        empty_bf16(1, 5),
        torch.empty(1, 1, device="cuda", dtype=torch.float32)
    ]
    register_replacement(rms_pattern_static, rms_replacement_static, inputs,
                         fwd_only, my_patterns)

    matches = []

    def record_match_fn(match: Match):
        matches.append(match)
        return False

    # with residual
    inputs = [
        empty_fp8(5, 4),
        empty_bf16(5, 4),
        empty_bf16(5, 4),
        empty_bf16(1, 5),
        torch.empty(1, 1, device="cuda", dtype=torch.float32)
    ]
    register_replacement(rms_pattern_residual_static,
                         rms_replacement_residual_static,
                         inputs,
                         fwd_only,
                         my_patterns,
                         extra_check=record_match_fn)

    return my_patterns, matches


def process_matches(matches, graph: torch.fx.Graph):
    for match in matches:
        nodes = list(graph.nodes)
        # TODO this is an expensive check
        if not all(node in nodes for node in match.nodes):
            raise ValueError(
                f"Broken match: not all nodes in graph: "
                f"{[node for node in match.nodes if node not in nodes]}")
        last_node_in_match = max(match.nodes, key=lambda x: nodes.index(x))
        with graph.inserting_after(last_node_in_match):
            kwargs = match.kwargs
            kwargs["epsilon"] = 1e-5

            fused_node = graph.call_function(
                auto_functionalized,
                (torch.ops._C.fused_add_rms_norm_static_fp8_quant.default, ),
                kwargs=kwargs)

            graph.inserting_after(fused_node)
            result_node_new = graph.call_function(operator.getitem,
                                                  (fused_node, 1))
            residual_node_new = graph.call_function(operator.getitem,
                                                    (fused_node, 2))

        def is_func(node, target):
            return node.op == "call_function" and node.target == target

        # find the output and the residual
        def find_auto_fn(op,
                         match=match):  # need to bind match to silence lint
            for node in match.nodes:
                if is_func(node, auto_functionalized) and node.args[0] == op:
                    return node
            return None

        def find_getitem(node, idx):
            for user in node.users:
                if is_func(node, operator.getitem) and user.args[1] == idx:
                    return user
            return None

        rms_node = find_auto_fn(torch.ops._C.fused_add_rms_norm.default)
        quant_node = find_auto_fn(torch.ops._C.static_scaled_fp8_quant.default)
        assert rms_node is not None
        assert quant_node is not None

        assert len(rms_node.users) == 2
        assert len(quant_node.users) == 1

        # meta["val"] is used by de-functionalization
        rms_val = rms_node.meta["val"]
        quant_val = quant_node.meta["val"]
        fused_node.meta["val"] = (None, quant_val[1], rms_val[1], rms_val[2])

        find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
        find_getitem(quant_node, 1).replace_all_uses_with(result_node_new)

    # Finally, remove matched nodes
    graph.eliminate_dead_code()
    assert all(node not in graph.nodes for match in matches
               for node in match.nodes)


def noop_pass(graph: torch.fx.Graph):
    pass


def get_fusion_pass():
    if not envs.VLLM_TORCH_COMPILE_FUSION:
        return noop_pass

    patterns, matches = get_patterns()

    def dump_graph(graph: torch.fx.Graph, stage: str):
        if stage in envs.VLLM_TORCH_COMPILE_FUSION_DUMP:
            logger.info("Printing graph to %s", f"{stage}.py")
            with open(f"{stage}.py", "w") as f:
                print(graph.python_code(root_module="self", verbose=True).src,
                      file=f)

    def fusion_pass(graph: torch.fx.Graph):
        """
        Use the pattern matcher
        """
        matches.clear()
        dump_graph(graph, "before_fusion")

        count = patterns.apply(graph)
        logger.info("Replaced %s patterns", count)
        dump_graph(graph, "after_pattern_match")

        # Manually process multi-output matches (and run DCE)
        process_matches(matches, graph)
        logger.info("Post-processed %s matches", len(matches))
        dump_graph(graph, "after_fusion")

    return fusion_pass
