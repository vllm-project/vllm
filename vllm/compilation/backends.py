import ast
import copy
import dataclasses
import os
import pprint
import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx
from typing import Tuple, List, Optional

import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

from .counter import compilation_counter
from .inductor_pass import InductorPass
from .monitor import end_monitoring_torch_compile
from .pass_manager import PostGradPassManager

logger = init_logger(__name__)

aten = torch.ops.aten

FILENO=0


def pprint(x):
    #print(x)
    pass


def match_gemm_rs_ag_gemm_orig():
    permute_2 = torch.ops.aten.permute(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce, tensor=mm_1, group_name='tp:0')
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=getitem_25, residual=getitem_1, weight=arg8_1, epsilon=1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]
    permute_3 = torch.ops.aten.permute(arg9_1, [1, 0])
    mm_2 = torch.ops.aten.mm(getitem_27, permute_3)
    return mm_2


def match_gemm_rs_ag_gemm_small(arg7_1, getitem_22):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    return getitem_25


def match_gemm_rs_ag_gemm_med(arg7_1, getitem_22, getitem_1, arg8_1):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = getitem_1, weight = arg8_1, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]
    return getitem_27, getitem_28
    #permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    #mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    #return mm_2


def match_gemm_rs_ag_gemm(arg7_1, getitem_22, arg8_1, getitem_1, arg9_1):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = getitem_1, weight = arg8_1, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]
    permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    return mm_2, getitem_28


# getitem_1 full residual
def replace_gemm_rs_ag_gemm_orig(arg7_1, getitem_24, arg8_1, getitem_1, arg9_1):
    split_1 = torch.ops.aten.split.Tensor(getitem_1, 2048)
    getitem_26 = split_1[0];  split_1 = None
    permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
    fused_matmul_reduce_scatter = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(getitem_24, clone, 'avg', 0, '0')
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = fused_matmul_reduce_scatter, residual = getitem_26, weight = arg8_1, epsilon = 1e-05)
    getitem_29 = auto_functionalized_4[1]
    getitem_30 = auto_functionalized_4[2]
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(getitem_1, getitem_30, 0, 0, 2048)
    split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, 2048)
    getitem_31 = split_2[0]  # local residual
    permute_5 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
    fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, '0')
    getitem_34 = fused_all_gather_matmul[1]
    getitem_35 = getitem_34[0]
    return getitem_35, getitem_31

def slices(residual) -> List[torch.Tensor]:
    n_slices = get_tensor_model_parallel_world_size()
    residual_slices = torch.chunk(residual, n_slices, dim=0)
    #pprint(f"SLICES {[r.shape for r in residual_slices]}")
    return residual_slices

@torch.library.custom_op("vllm::gemm_rs_ag_gemm", mutates_args=())
def gemm_rs_ag_gemm(arg7_1: torch.Tensor, getitem_22: torch.Tensor, arg8_1: torch.Tensor, getitem_1: torch.Tensor, arg9_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_name = torch.distributed.group.WORLD.group_name # TODO: factor out to setup

    # this is terrible
    if True:
        res_slices = slices(getitem_1)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
        print(f"SLICE_SIZE = {slice_size}, orig_shape={getitem_1.shape}, slice_shapes=[{[x.shape for x in res_slices]}]")
    else:
        slice_size = 2048

    split_1 = torch.ops.aten.split.Tensor(getitem_1, slice_size)  # XXXXXXXXXXX
    getitem_26 = split_1[0];  split_1 = None
    permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
    fused_matmul_reduce_scatter = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(getitem_22, clone, 'avg', 0, group_name) # XXXXXXXXXXXXX
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = fused_matmul_reduce_scatter, residual = getitem_26, weight = arg8_1, epsilon = 1e-05)
    getitem_29 = auto_functionalized_4[1]
    getitem_30 = auto_functionalized_4[2]
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(getitem_1, getitem_30, 0, 0, slice_size) # XXXXXXXXXXXXXXX
    split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, slice_size) # XXXXXXXXXXX
    getitem_31 = split_2[0]  # local residual
    permute_5 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
    fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, group_name) # XXXXXXXXXXX
    getitem_34 = fused_all_gather_matmul[1]
    getitem_35 = getitem_34[0]
    return getitem_35, getitem_31   # matmul, residual

# this is wrong
@torch.library.register_fake("vllm::gemm_rs_ag_gemm")
def gemm_rs_ag_gemm_fake(arg7_1: torch.Tensor, getitem_22: torch.Tensor, arg8_1: torch.Tensor, getitem_1: torch.Tensor, arg9_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mm_res = torch.empty((getitem_22.shape[0], arg9_1.shape[0]), device=getitem_22.device, dtype=getitem_22.dtype)  #???
    resid = torch.empty_like(getitem_1)
    return (mm_res, resid)

def replace_gemm_rs_ag_gemm(arg7_1, getitem_22, arg8_1, getitem_1, arg9_1):
    results = torch.ops.vllm.gemm_rs_ag_gemm(arg7_1, getitem_22, arg8_1, getitem_1, arg9_1)
    getitem_34 = results[0]
    getitem_35 = results[1]
    return getitem_34, getitem_35


def match_final(arg227_1, getitem_1022, getitem_1020, arg228_1):
    permute_128 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_127 = torch.ops.aten.mm.default(getitem_1022, permute_128)
    auto_functionalized_224 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_127, group_name = 'tp:0') # TODO: not same as group name
    getitem_1024 = auto_functionalized_224[1]
    auto_functionalized_225 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1024, residual = getitem_1020, weight = arg228_1, epsilon = 1e-05)
    getitem_1026 = auto_functionalized_225[1]
    return getitem_1026


def replace_final(arg227_1, getitem_1215, getitem_1209, arg228_1):
    group_name = torch.distributed.group.WORLD.group_name # factor out?
    permute_254 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_1215, permute_254)
    auto_functionalized_161 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0') # TODO: not same as group name
    getitem_1217 = auto_functionalized_161[1]
    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_1209, 2, group_name)
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor)
    auto_functionalized_162 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1217, residual = wait_tensor, weight = arg228_1, epsilon = 1e-05)
    getitem_1219 = auto_functionalized_162[1]
    return getitem_1219


my_patterns: Optional[PatternMatcherPass] = None
my_patterns2: Optional[PatternMatcherPass] = None

def get_matches():
    global my_patterns
    global my_patterns2
    matches = []
    matches2 = []

    def record_match_fn(match: Match):
        matches.append(match)
        return False

    def record_match_fn2(match: Match):
        matches2.append(match)
        return False

    if not my_patterns:
        my_patterns = PatternMatcherPass()
        my_patterns2 = PatternMatcherPass()

        x = torch.empty([4,4], device='cuda')
        w = torch.empty([4,4], device='cuda')
        resid = torch.empty([4,4], device='cuda')
        resid_w = torch.empty([4,4], device='cuda')
        x2 = torch.empty([4,4], device='cuda')
        inputs = [x, w, resid, resid_w, x2]
        inputs_small = inputs[0:2]
        inputs_med = inputs[0:4]

        register_replacement(match_gemm_rs_ag_gemm,
                             replace_gemm_rs_ag_gemm,
                             inputs,
                             fwd_only,
                             [my_patterns],
                             extra_check=record_match_fn)

        final_inputs = [x, w, resid, resid_w]
        register_replacement(match_final,
                             replace_final,
                             final_inputs,
                             fwd_only,
                             [my_patterns2])

    return matches, matches2

def process_matches(graph: fx.Graph, matches):
    print(f"len = {len(matches)}")
    for match in matches:
        last_node_in_match = match.nodes[-1] #max(match.nodes, key=lambda x: nodes.index(x))

        with graph.inserting_after(last_node_in_match):
            fused_node = graph.call_function(torch.ops.vllm.gemm_rs_ag_gemm.default, kwargs=match.kwargs)

            graph.inserting_after(fused_node)
            result_node_new = graph.call_function(operator.getitem, (fused_node, 0))
            residual_node_new = graph.call_function(operator.getitem, (fused_node, 1))

        # find the output and the residual
        def find_fn(op):
            for node in reversed(match.nodes):
                if node.op == "call_function" and node.target == op:
                    return node
            return None

        def find_auto_fn(op):
            for node in reversed(match.nodes):
                if node.op == "call_function" and node.target == auto_functionalized and node.args[0] == op:
                    return node
            return None

        def find_getitem(node, idx):
            for user in reversed(node.users):
                if user.op == "call_function" and user.target == operator.getitem and user.args[1] == idx:
                    return user
            return None

        rms_node = find_auto_fn(torch.ops._C.fused_add_rms_norm.default)
        gemm_node = find_fn(torch.ops.aten.mm.default)
        assert rms_node is not None
        assert gemm_node is not None

        #assert len(rms_node.users) == 2
        #assert len(gemm_node.users) == 1

        # meta["val"] is used by de-functionalization
        rms_val = rms_node.meta["val"]
        gemm_val = gemm_node.meta["val"]
        fused_node.meta["val"] = (gemm_val, rms_val[2])

        find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
        gemm_node.replace_all_uses_with(result_node_new)

    # Finally, remove matched nodes
    graph.eliminate_dead_code()
    assert all(node not in graph.nodes for match in matches for node in match.nodes)

def process_matches2(graph: fx.Graph, matches2):
    print(f"len2 = {len(matches2)}")

def async_rewrite(graph: fx.Graph):
    matches, matches2 = get_matches()
    matches.clear()
    matches2.clear()

    count = my_patterns.apply(graph)
    print(f"fused gemm match count = {count}")
    count = my_patterns2.apply(graph)
    print(f"final match count = {count}")

    process_matches(graph, matches)
    process_matches2(graph, matches2)

    return graph


class InductorHashCache:
    """
    Disk format: a Python list of tuples, each tuple is
    (runtime_shape, graph_index, hash_str)
    We use list of tuple for readability.

    In-memory format: a defaultdict of dict, where the key is
    runtime_shape, and the value is a dict of graph_index to hash_str.

    The data is essentially `Dict[Optional[int], Dict[int, str]]`,
    we don't use json here because json doesn't support int as key.

    TODO: better off-the-shelf solution to serialize the data?
    """

    def __init__(self, cache_dir: str, disabled: bool = False):
        self.cache: defaultdict = defaultdict(dict)
        self.disabled = disabled
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir,
                                            "inductor_hash_cache.py")
        if disabled:
            return
        # set flags so that Inductor and Triton store their cache
        # in the cache_dir, then users only need to copy the cache_dir
        # to another machine to reuse the cache.
        inductor_cache = os.path.join(cache_dir, "inductor_cache")
        os.makedirs(inductor_cache, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        triton_cache = os.path.join(cache_dir, "triton_cache")
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.deserialize(f.read())

    def deserialize(self, data: str):
        # we use ast.literal_eval to parse the data
        # because it is a safe way to parse Python literals.
        # do not use eval(), it is unsafe.
        list_data = ast.literal_eval(data)
        for runtime_shape, graph_index, hash_str in list_data:
            self.cache[runtime_shape][graph_index] = hash_str

    def serialize(self) -> str:
        data = []
        for runtime_shape, graph_index_to_hash_str in self.cache.items():
            for graph_index, hash_str in graph_index_to_hash_str.items():
                data.append((runtime_shape, graph_index, hash_str))
        printer = pprint.PrettyPrinter(indent=4)
        return printer.pformat(data)

    def save_to_file(self):
        if self.disabled:
            return
        with open(self.cache_file_path, "w") as f:
            f.write(self.serialize())

    def __contains__(self, key: Tuple[Optional[int], int]) -> bool:
        if self.disabled:
            return False
        runtime_shape, graph_index = key
        return runtime_shape in self.cache and graph_index in self.cache[
            runtime_shape]

    def __getitem__(self, key: Tuple[Optional[int], int]) -> str:
        if self.disabled:
            raise KeyError("cannot read from disabled cache")
        runtime_shape, graph_index = key
        return self.cache[runtime_shape][graph_index]

    def __setitem__(self, key: Tuple[Optional[int], int], value: str):
        # setitem for disabled cache is fine, because we
        # don't actually write to the disk
        runtime_shape, graph_index = key
        self.cache[runtime_shape][graph_index] = value


class AlwaysHitShapeEnv:
    """
    Why do we need this class:

    For normal `torch.compile` usage, every compilation will have
    one Dynamo bytecode compilation and one Inductor compilation.
    The Inductor compilation happens under the context of the
    Dynamo bytecode compilation, and that context is used to
    determine the dynamic shape information, etc.

    For our use case, we only run Dynamo bytecode compilation once,
    and run Inductor compilation multiple times with different shapes
    plus a general shape. The compilation for specific shapes happens
    outside of the context of the Dynamo bytecode compilation. At that
    time, we don't have shape environment to provide to Inductor, and
    it will fail the Inductor code cache lookup.

    By providing a dummy shape environment that always hits, we can
    make the Inductor code cache lookup always hit, and we can
    compile the graph for different shapes as needed.

    The following dummy methods are obtained by trial-and-error
    until it works.
    """

    def __init__(self) -> None:
        self.guards: List[Any] = []

    def evaluate_guards_expression(self, *args, **kwargs):
        return True

    def get_pruned_guards(self, *args, **kwargs):
        return []

    def produce_guards_expression(self, *args, **kwargs):
        return ""


def wrap_inductor(graph: fx.GraphModule,
                  example_inputs,
                  additional_inductor_config,
                  compilation_config: CompilationConfig,
                  graph_index: int = 0,
                  num_graphs: int = 1,
                  runtime_shape: Optional[int] = None,
                  use_inductor: bool = True) -> Any:
    if graph_index == 0:
        # before compiling the first graph, record the start time
        global compilation_start_time
        compilation_start_time = time.time()

    if not use_inductor:
        return graph

    compilation_counter.num_inductor_compilations += 1

    from torch._inductor import config
    torch._inductor.config._micro_pipeline_tp = True
    current_config = config.get_config_copy()
    from torch._inductor.compile_fx import compile_fx

    if additional_inductor_config is not None:
        current_config.update(additional_inductor_config)

    if isinstance(runtime_shape, int):
        # for a specific batchsize, tuning triton kernel parameters
        # can be beneficial
        current_config["max_autotune"] = True
        current_config["coordinate_descent_tuning"] = True

    # inductor can inplace modify the graph, so we need to copy it
    # see https://github.com/pytorch/pytorch/issues/138980
    graph = copy.deepcopy(graph)

    cache_data = compilation_config.inductor_hash_cache
    if (runtime_shape, graph_index) in cache_data:
        # we compiled this graph before
        # so we can directly lookup the compiled graph via hash
        hash_str = cache_data[(runtime_shape, graph_index)]
        if graph_index == 0:
            # adds some info logging for the first graph
            logger.info(
                "Directly lookup the graph for shape %s from the cache",
                str(runtime_shape))  # noqa
        logger.debug(
            "directly lookup the %s-th graph for shape %s via hash %s",
            graph_index, str(runtime_shape), hash_str)
        from torch._inductor.codecache import FxGraphCache
        with patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                   lambda *args, **kwargs: AlwaysHitShapeEnv()):
            inductor_compiled_graph = FxGraphCache._lookup_graph(
                hash_str, example_inputs, True, False)
            assert inductor_compiled_graph is not None, (
                "Inductor cache lookup failed. Please remove"
                f"the cache file {compilation_config.inductor_hash_cache.cache_file_path} and try again."  # noqa
            )

        # Inductor calling convention (function signature):
        # f(list) -> tuple
        # Dynamo calling convention (function signature):
        # f(*args) -> Any

        # need to know if the graph returns a tuple
        from torch._inductor.compile_fx import graph_returns_tuple
        returns_tuple = graph_returns_tuple(graph)

        # this is the callable we return to Dynamo to run
        def compiled_graph(*args):
            # convert args to list
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            # unpack the tuple if needed
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]
    else:
        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.
        from torch._inductor.codecache import compiled_fx_graph_hash

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            out = compiled_fx_graph_hash(*args, **kwargs)
            # store the hash in the cache
            nonlocal cache_data
            cache_data[(runtime_shape, graph_index)] = out[0]
            if graph_index == 0:
                # adds some info logging for the first graph
                logger.info("Cache the graph of shape %s for later use",
                            str(runtime_shape))
            logger.debug("store the %s-th graph for shape %s via hash %s",
                         graph_index, str(runtime_shape), out[0])
            return out

        def _check_can_cache(*args, **kwargs):
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For vLLM, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        with patch(# for hijacking the hash of the compiled graph
                "torch._inductor.codecache.compiled_fx_graph_hash",
                hijack_compiled_fx_graph_hash), \
            patch(# for providing a dummy shape environment
                "torch._inductor.codecache.FxGraphCache._get_shape_env",
                 _get_shape_env), \
            patch(# for forcing the graph to be cached
                "torch._inductor.codecache.FxGraphCache._check_can_cache",
                _check_can_cache):
            compiled_graph = compile_fx(graph,
                                        example_inputs,
                                        config_patches=current_config)

    # after compiling the last graph, record the end time
    if graph_index == num_graphs - 1:
        now = time.time()
        elapsed = now - compilation_start_time
        compilation_config.compilation_time += elapsed
        if runtime_shape is None:
            logger.info("Compiling a graph for general shape takes %.2f s",
                        elapsed)
        else:
            logger.info("Compiling a graph for shape %s takes %.2f s",
                        runtime_shape, elapsed)

    return compiled_graph


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(graph: fx.GraphModule,
                ops: List[str]) -> Tuple[fx.GraphModule, List[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(self, module: torch.fx.GraphModule,
                 compile_submod_names: List[str], vllm_config: VllmConfig,
                 graph_pool):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.vllm_config = vllm_config

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode:
            return super().run(*fake_args)

    def call_module(self, target: torch.fx.node.Target,
                    args: Tuple[torch.fx.node.Argument,
                                ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            global compilation_start_time
            compiled_graph_for_general_shape = wrap_inductor(
                submod,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=index,
                num_graphs=len(self.compile_submod_names),
                runtime_shape=None,
                use_inductor=self.compilation_config.use_inductor)

            self.module.__dict__[target] = PiecewiseBackend(
                submod, self.vllm_config, self.graph_pool, index,
                len(self.compile_submod_names), sym_shape_indices,
                compiled_graph_for_general_shape)

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class VllmBackend:
    """The compilation backend for `torch.compile` with VLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: List[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: List[int]
    input_buffers: List[torch.Tensor]

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def configure_post_pass(self):
        config = self.compilation_config
        self.post_grad_pass_manager.configure(config.pass_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        inductor_config = config.inductor_compile_config
        PASS_KEY = "post_grad_custom_post_pass"
        if PASS_KEY in inductor_config:
            # Config should automatically wrap all inductor passes
            assert isinstance(inductor_config[PASS_KEY], InductorPass)
            self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .monitor import torch_compile_start_time
        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Dynamo bytecode transform time: %.2f s", dynamo_time)
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_config.splitting_ops)

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(
            self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.vllm_config,
                                    self.graph_pool).run(*example_inputs)

        self._called = True

        if not self.compilation_config.use_cudagraph or \
            not self.compilation_config.cudagraph_copy_inputs:
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]

        # this is the callable we return to Dynamo to run
        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]

                # copy the tensor to the static buffer
                static_tensor.copy_(runtime_tensor)

                # replace the tensor in the list_args to the static buffer
                list_args[index] = static_tensor
            return self.split_gm(*list_args)

        return copy_and_call


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[List[int]] = None


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: List[int],
                 compiled_graph_for_general_shape: Callable):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: Set[int] = set(
            self.compilation_config.compile_sizes)
        self.capture_sizes: Set[int] = set(
            self.compilation_config.capture_sizes
        ) if self.compilation_config.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: Set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.compilation_config.inductor_hash_cache.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = wrap_inductor(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
                use_inductor=self.compilation_config.use_inductor)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
                        runtime_shape)
                return entry.runnable(*args)

            if self.is_first_graph:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_caputured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
