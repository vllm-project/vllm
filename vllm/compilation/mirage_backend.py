import os
from collections import defaultdict
import time
from mirage import MPK, MPKMetadata, MirageModelConfig
import re
from typing import Any

import torch
import torch.fx as fx

from vllm.config import CompilationConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.logger import init_logger

from .counter import compilation_counter

logger = init_logger(__name__)

# TODO(Jianan Ji): Is this name mapping common for all models?
def transfer_tensor_names(placeholders: list[torch.fx.node.Node]) -> list[str]:
    """Transfer FX placeholder debug names to model-like dotted names. Return a list of transferred names and input id.

    Example:
        l_self_modules_layers_modules_17_modules_mlp_\
            modules_gate_up_proj_parameters_weight_
        -> model.layers.17.mlp.gate_up_proj.weight

    Notes:
    - Tailored for Qwen3-style module names seen in exported FX graphs.
    - We do NOT rename the FX node identifiers (dots are not valid in FX names).
      Instead, we annotate via node.meta['logical_name'] and return the list.
    """
    converted_names = []
    s_pattern = re.compile(r"^s\d+$") # s72 / s80

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

def build_model_config(
    model_config: ModelConfig,
    state_dict: dict[str, torch.Tensor],
    k_cache_tensors: list[torch.Tensor],
    v_cache_tensors: list[torch.Tensor],
    position_embeddings_: torch.Tensor,
    parallel_config: ParallelConfig,
) -> MirageModelConfig:
    whole_dim = position_embeddings_.shape[-1]
    cos_tensor_ = position_embeddings_[:, 0:whole_dim//2].unsqueeze(0)
    sin_tensor_ = position_embeddings_[:, whole_dim//2:].unsqueeze(0)
    
    cos_tensor = torch.cat([cos_tensor_, cos_tensor_], dim=-1)
    sin_tensor = torch.cat([sin_tensor_, sin_tensor_], dim=-1)
    
    position_embeddings = (cos_tensor, sin_tensor)
    mirage_model_config = MirageModelConfig(
        # model architecture
        hidden_size=model_config.get_hidden_size(),
        intermediate_size=getattr(model_config.hf_text_config, "intermediate_size", 0),
        vocab_size=model_config.get_vocab_size(),
        local_num_q_heads=model_config.get_num_attention_heads(parallel_config),
        local_num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_dim=model_config.get_head_size(),
        num_layers=getattr(model_config.hf_text_config, "num_hidden_layers", 0),
        # kv cache
        k_cache=k_cache_tensors,
        v_cache=v_cache_tensors,
        # position embeddings
        position_embeddings=position_embeddings,
        # model weights
        state_dict=state_dict,
        with_lm_head=False,
    )
    return mirage_model_config

def build_mpk_metadata(
        vllm_config: VllmConfig, 
        args: list[Any],
        transfered_tensor_names: list[str],
    ) -> MPKMetadata:
    forward_context = get_forward_context()
    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    # For now we assume only one attention group
    attn_metadata = list(forward_context.attn_metadata.values())[0]

    static_forward_context = forward_context.no_compile_layers # layer names to layers
    k_cache_tensors = []
    v_cache_tensors = []
    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in static_forward_context.keys():
        index2name[extract_layer_index(layer_name, 1)].append(layer_name)

    for layer_index in sorted(index2name):
        layer_names = index2name[layer_index]
        assert len(layer_names) == 1, "Multiple layers with the same layer index are not supported"
        layer_name = layer_names[0]
        k_cache_tensors.append(static_forward_context[layer_name].kv_cache[0][0])
        v_cache_tensors.append(static_forward_context[layer_name].kv_cache[0][1])
    
    state_dict = {}
    input_token_ids = None
    positions_tensor = None
    position_embeddings = None
    for arg, name in zip(args, transfered_tensor_names):
        if name == 'input_ids':
            input_token_ids = arg
        elif name == 'positions':
            positions_tensor = arg
        elif "cos_sin_cache" in name:
            position_embeddings = arg
        elif "qkv" in name: 
            # Split qkv since we need to shuffle them on mirage side later
            # (6144, 4096) -> (4096, 4096), (1024, 4096), (1024, 4096)
            qkv_tensor = arg

            total_dim = qkv_tensor.shape[0]
            n_q_heads = model_config.get_num_attention_heads(parallel_config) # 32
            n_kv_heads = model_config.get_num_kv_heads(parallel_config) # 8
            n_heads = n_q_heads + n_kv_heads * 2
            
            q_range = (total_dim * n_q_heads) // n_heads # 6144 * 32 / 48 = 4096
            k_range = (total_dim * (n_q_heads + n_kv_heads)) // n_heads # 6144 * 40 / 48 = 5120

            q_tensor = qkv_tensor[:q_range, :]
            k_tensor = qkv_tensor[q_range:k_range, :]
            v_tensor = qkv_tensor[k_range:, :]
            
            # substitute qkv to q/k/v views
            state_dict[name.replace("qkv", "q")] = q_tensor
            state_dict[name.replace("qkv", "k")] = k_tensor
            state_dict[name.replace("qkv", "v")] = v_tensor
            
            state_dict[name] = qkv_tensor
        elif "gate_up" in name:
            # Split gate_up to gate and up
            gate_up_tensor = arg
            total_dim = gate_up_tensor.shape[0]
            single_dim = total_dim // 2
            
            gate_tensor = gate_up_tensor[:single_dim, :]
            up_tensor = gate_up_tensor[single_dim:, :]
            
            # substitude gate_up to gate and up
            state_dict[name.replace("gate_up", "gate")] = gate_tensor
            state_dict[name.replace("gate_up", "up")] = up_tensor
            
            state_dict[name] = gate_up_tensor
        else:
            state_dict[name] = arg
    
    mirage_model_config = build_model_config(
        model_config,
        state_dict,
        k_cache_tensors,
        v_cache_tensors,
        position_embeddings,
        parallel_config,
    )
    mpk_metadata = MPKMetadata(
        mode = "online_notoken",
        # total_num_requests
        # num_remote_schedulers: int = 0
        max_seq_length = model_config.max_model_len,
        max_num_batched_requests = scheduler_config.max_num_seqs,
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens,
        max_num_pages = cache_config.num_gpu_blocks,
        page_size = cache_config.block_size,
        # max_sm_num: int = 108
        device = "cuda",
        # # model 
        weight_from_model = False,
        model_name = model_config.model,
        # model_path: Optional[str] = None
        # multi device support
        world_size = parallel_config.world_size,
        rank = parallel_config.rank,
        # # Meta tensors
        step = positions_tensor,
        # tokens: Optional[torch.Tensor] = None
        input_tokens = input_token_ids,
        # output_tokens: Optional[torch.Tensor] = None
        # num_new_tokens: Optional[torch.Tensor] = None
        # prompt_lengths: Optional[torch.Tensor] = None
        qo_indptr_buffer = attn_metadata.qo_indptr_gpu,
        paged_kv_indptr_buffer = attn_metadata.paged_kv_indptr_gpu,
        paged_kv_indices_buffer = attn_metadata.paged_kv_indices_gpu,
        paged_kv_last_page_len_buffer = attn_metadata.paged_kv_last_page_len_gpu,
        # kv cache tensors, weights and model config
        model_config=mirage_model_config,
        # # profiling
        # profiler_tensor: Optional[torch.Tensor] = None
        # trace_name: Optional[str] = None
        # # spec decode config
        # spec_decode_config: Optional[object] = None
    )
    return mpk_metadata

class MirageBackend:
    """The compilation backend for Mirage Persistent Kernel."""

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        logger.debug("[Mirage] Calling MirageBackend init!")

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.model_config = vllm_config.model_config
        self.model_name = vllm_config.model_config.model

    def __call__(
        self, graph: fx.GraphModule, example_inputs
    ) -> Any:

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .monitor import torch_compile_start_time
        
        # TODO: remove this after debugging
        # try:
        #     src = graph.print_readable(print_output=False)
        # except Exception:
        #     src = str(graph)
        # try:
        #     with open('mirage_backends_graph.txt', 'w') as f:
        #         logger.info('Writing readable FX graph to mirage_backends_graph.txt')
        #         f.write(src)
        #         logger.info('Readable FX graph written to mirage_backends_graph.txt')
        # except Exception:
        #     logger.exception('Failed to write mirage_backends_graph.txt')

        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Dynamo bytecode transform time: %.2f s", dynamo_time)
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "MirageBackend can only be called once"
        
        placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
        assert len(placeholders) == len(example_inputs)
        
        transfered_tensor_names = transfer_tensor_names(placeholders)

        max_input_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        
        # TODO(Jianan Ji): remove this after debugging
        # with open('mirage_backends_graph.txt', 'w') as f:
        #     f.write(graph.print_readable(print_output=False))
        # with open("graph_structure.txt", "w", encoding="utf-8") as f:
        #     f.write(str(graph.graph))
        
        
        self._called = True
        self.compiled = False
        
        def compile_or_call(*args):
            dumb_run_called = (get_forward_context().attn_metadata is None)
            if dumb_run_called:
                model_config = self.vllm_config.model_config
                dtype = model_config.dtype
                hidden_size = model_config.get_hidden_size()
                # # TODO(Jianan Ji): We'll want to run graph(*args) instead of doing nothing
                output_tensor = torch.zeros(2, hidden_size, device='cuda', dtype=dtype)
                # logger.info(f"[Mirage] Calling dumb_run_called, returning dummy output tensor with shape [{output_tensor.shape}]......!")

                return (output_tensor,)
                # return graph(*args)
            
            if not self.compiled:
                # Compile only at the first call -- when we get real tensors
                logger.info("[Mirage] Calling compile_or_call for the first time, compiling......!")
                mpk_metadata = build_mpk_metadata(
                    self.vllm_config,
                    args,
                    transfered_tensor_names,
                )
                logger.info(f"[Mirage] MPK metadata: {mpk_metadata.info_as_string()}")
                self.mpk = MPK(mpk_metadata)
                self.mpk.build()
                self.mpk.compile(output_dir=os.path.join(os.path.dirname(__file__), "mirage_backend_output"))
                
                self.compiled = True
                
            result_hidden_states = self.mpk()
            
            return (result_hidden_states,)
        
        return compile_or_call