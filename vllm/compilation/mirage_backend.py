from collections import defaultdict
from .backends import *
from mirage.mpk import MPK, MPKMetadata
import re
from vllm.config import get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
import torch
def transfer_tensor_names(placeholders: list[torch.fx.node.Node]) -> list[str]:
    """Transfer FX placeholder debug names to model-like dotted names.

    Example:
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_up_proj_parameters_weight_
        -> model.layers.17.mlp.gate_up_proj.weight

    Notes:
    - Tailored for Qwen3-style module names seen in exported FX graphs.
    - We do NOT rename the FX node identifiers (dots are not valid in FX names).
      Instead, we annotate via node.meta['logical_name'] and return the list.
    """
    converted_names = []
    s_pattern = re.compile(r"^s\d+$")

    for node in placeholders:
        name = node.name
        if name == 'l_input_ids_':
            final_name = 'input_ids'
            converted_names.append(final_name)
        elif name == 'l_positions_':
            final_name = 'positions'
            converted_names.append(final_name)
        elif s_pattern.match(name): # s72 / s80
            converted_names.append(name)
        else:
            if name.startswith('l_self_modules_'):
                name = name.replace('l_self_modules_', '', 1)
            if name.endswith('_'):
                name = name[:-1]

            name = name.replace('_modules_', '.')
            name = name.replace('_parameters_', '.')

            final_name = 'model.' + name

            converted_names.append(final_name)

    return converted_names


# @dataclass
# class MPKMetadata:
#     # ---------- MPK class external state bundled here ----------
#     # args
#     mode: str = "offline"
#     total_num_requests: int = 1
#     num_remote_schedulers: int = 0
#     max_seq_length: int = 0
#     max_num_batched_requests: int = 0
#     max_num_batched_tokens: int = 0
#     max_num_pages: int = 0
#     page_size: int = 0
#     max_sm_num: int = 108
#     device: str = "cuda"
#     # model 
#     weight_from_model: bool
#     model_name: Optional[str] # For now, model_name must be provided
#     model_path: Optional[str] = None
#     # fx graph
#     state_dict: Optional[dict] = None
#     # Meta tensors
#     step: Optional[torch.Tensor] = None
#     tokens: Optional[torch.Tensor] = None
#     input_tokens: Optional[torch.Tensor] = None
#     output_tokens: Optional[torch.Tensor] = None
#     num_new_tokens: Optional[torch.Tensor] = None
#     prompt_lengths: Optional[torch.Tensor] = None
#     qo_indptr_buffer: Optional[torch.Tensor] = None
#     paged_kv_indptr_buffer: Optional[torch.Tensor] = None
#     paged_kv_indices_buffer: Optional[torch.Tensor] = None
#     paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None
#     # profiling
#     profiler_tensor: Optional[torch.Tensor] = None
#     trace_name: Optional[str] = None
#     # spec decode config
#     spec_decode_config: Optional[object] = None
    

def build_mpk_metadata(
        vllm_config: VllmConfig, 
        forward_context: ForwardContext,
        state_dict: dict[str, torch.Tensor],
        k_cache_tensors: list[torch.Tensor],
        v_cache_tensors: list[torch.Tensor]
    ) -> MPKMetadata:
    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    attn_metadata = forward_context.attn_metadata
    mpk_metadata = MPKMetadata(
        mode = "online"
        # total_num_requests
        # num_remote_schedulers: int = 0
        max_seq_length = model_config.max_model_len,
        max_num_batched_requests = scheduler_config.max_num_seqs,
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens,
        max_num_pages = cache_config.num_gpu_blocks
        page_size = cache_config.block_size
        # max_sm_num: int = 108
        device: str = "cuda"
        # # model 
        # weight_from_model: bool
        model_name = model_config.model_name,
        # model_path: Optional[str] = None
        # multi device support
        world_size = parallel_config.world_size
        rank = parallel_config.rank
        # # fx graph
        state_dict = state_dict
        # # Meta tensors
        # step: Optional[torch.Tensor] = None
        # tokens: Optional[torch.Tensor] = None
        # input_tokens: Optional[torch.Tensor] = None
        # output_tokens: Optional[torch.Tensor] = None
        # num_new_tokens: Optional[torch.Tensor] = None
        # prompt_lengths: Optional[torch.Tensor] = None
        qo_indptr_buffer = attn_metadata.qo_indptr_gpu
        paged_kv_indptr_buffer = attn_metadata.paged_kv_indptr_gpu
        paged_kv_indices_buffer = attn_metadata.paged_kv_indices_gpu
        paged_kv_last_page_len_buffer = attn_metadata.paged_kv_last_page_len_gpu
        # kv cache tensors
        k_cache_tensors = k_cache_tensors
        v_cache_tensors = v_cache_tensors
        # # profiling
        # profiler_tensor: Optional[torch.Tensor] = None
        # trace_name: Optional[str] = None
        # # spec decode config
        # spec_decode_config: Optional[object] = None
    )
    return mpk_metadata

class MirageBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. language_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.model_name = vllm_config.model_config.model_name
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config
        )

    def __call__(
        self, graph: fx.GraphModule, example_inputs
    ) -> VllmSerializableFunction:

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .monitor import torch_compile_start_time

        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Dynamo bytecode transform time: %.2f s", dynamo_time)
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "MirageBackend can only be called once"
        
        placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
        assert len(placeholders) == len(example_inputs)
        
        transfered_tensor_names = transfer_tensor_names(placeholders)
        
        forward_context = get_forward_context()
        static_forward_context = forward_context.no_compile_layers # layer names to layers
        
        k_cache_tensors = []
        v_cache_tensors = []
        # Convert kv_caches dict to a list of tensors in the order of layer_index.
        index2name = defaultdict(list)
        for layer_name in static_forward_context.keys():
            index2name[extract_layer_index(layer_name, 1)].append(layer_name)

        for layer_index in sorted(index2name.keys()):
            layer_names = index2name[layer_index]
            assert len(layer_names) == 1, "Multiple layers with the same layer index are not supported"
            layer_name = layer_names[0]
            k_cache_tensors.append(static_forward_context[layer_name].kv_cache[0])
            v_cache_tensors.append(static_forward_context[layer_name].kv_cache[1])
            # kv_cache_tensors shape: num_layers * (2, num_blocks, block_size, num_kv_heads, head_size)

        self._called = True
        self.compiled = False
        
        def compile_or_call(*args):
            if not self.compiled:
                # Compile only at the first call -- when we get real tensors
                state_dict = {}
                for arg, name in zip(args, transfered_tensor_names):
                    if name == 'input_ids':
                        input_tensor = arg
                    elif name == 'positions':
                        positions_tensor = arg
                    else:
                        state_dict[name] = arg
                vllm_config = get_current_vllm_config()

                model_config = vllm_config.model_config
                mpk_metadata = build_mpk_metadata(
                    vllm_config,
                    forward_context,
                    state_dict,
                    k_cache_tensors,
                    v_cache_tensors
                )
                self.mpk = MPK(mpk_metadata)
                self.mpk.build()
                self.mpk.compile()
                
                self.compiled = True
                
            return self.mpk()
        
        return VllmSerializableFunction(
            graph, example_inputs, self.prefix, compile_or_call
        )