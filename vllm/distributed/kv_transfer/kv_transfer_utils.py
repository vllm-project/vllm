import hashlib
import threading

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_kv_transfer_group

# Thread-local storage for context management
_hook_context = threading.local()


# the built-in hash function is not deterministic
# due to the design of hash randomization.
# so we need a way to hash the tensor in a deterministic way
# between the Prefill & Decode processes.
def get_tensor_stable_hash(tensor):
    data = tensor.cpu().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


def set_context_value(value):
    """Set a value in the thread-local context."""
    _hook_context.value = value


def get_context_value():
    """Get the value from the thread-local context."""
    return getattr(_hook_context, "value", None)


def maybe_register_PD_disagg_hooks(model: torch.nn.Module,
                                   vllm_config: VllmConfig):
    if vllm_config.kv_transfer_config is None:
        return

    if (vllm_config.kv_transfer_config.is_kv_producer
            and vllm_config.kv_transfer_config.is_layerwise_kv_transfer):
        register_model_PD_disagg_hooks(model, vllm_config)
        register_decoder_layer_PD_disagg_hooks(model)


def register_model_PD_disagg_hooks(model: torch.nn.Module,
                                   vllm_config: VllmConfig):
    """
    Registers hooks:
    - A pre-forward hook to save context data.
    - A post-forward hook to send the hidden states.
    """

    # Pre-forward hook for the top-level model
    def pre_forward_hook(module, args, kwargs):
        if 'input_ids' not in kwargs:
            raise ValueError("No input_ids tensor found in kwargs.")
        if 'attn_metadata' not in kwargs:
            raise ValueError("No attn_metadata tensor found in kwargs.")

        input_ids = kwargs['input_ids']
        attn_metadata = kwargs['attn_metadata']

        # skip if it is a profile run
        if input_ids.view(-1)[0].item() == 0:
            return

        input_id_hashes = []
        start_pos = 0
        for seq_length in attn_metadata.seq_lens:
            end_pos = start_pos + seq_length
            input_id_hashes.append(
                get_tensor_stable_hash(input_ids[start_pos:end_pos]))
            start_pos = end_pos

        context_dict = {
            'input_id_hashes': input_id_hashes,
            'block_size': vllm_config.cache_config.block_size
        }
        set_context_value(context_dict)

    def post_forward_hook(module, args, kwargs, output):

        # in case of PP, the output might be of IntermediateTensors type
        if not isinstance(output, torch.Tensor):
            return output

        context = get_context_value()
        if context is None or 'input_id_hashes' not in context:
            return output

        input_id_hashes = get_context_value()['input_id_hashes']

        if 'attn_metadata' not in kwargs:
            raise ValueError("No attn_metadata tensor found in kwargs.")

        attn_metadata = kwargs['attn_metadata']

        hidden_states = output

        get_kv_transfer_group().send_hidden_states(input_id_hashes,
                                                   hidden_states,
                                                   attn_metadata)

        return output

    # Register pre-forward and post-forward hooks to the top-level model
    model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
    model.register_forward_hook(post_forward_hook, with_kwargs=True)


def register_decoder_layer_PD_disagg_hooks(module: torch.nn.Module,
                                           suffix="DecoderLayer"):
    """
    Find the modules of decoder layers and register forward hooks to send
    kv cache of one layer.
    """

    def create_decoderlayer_hook(idx):

        def decoderlayer_hook(module, args, kwargs, output):
            # do nothing if is it is profile run
            context = get_context_value()
            if context is None:
                return output
            if any(key not in context
                   for key in ('input_id_hashes', 'block_size')):
                return output

            input_id_hashes = context['input_id_hashes']
            block_size = context['block_size']

            kv_cache, attn_metadata = args[2], args[3]
            get_kv_transfer_group().send_one_layer_kv_cache(
                idx, input_id_hashes, kv_cache, attn_metadata, block_size)

            return output

        return decoderlayer_hook

    if hasattr(module, "layers") and isinstance(
            module.layers,
        (list, torch.nn.ModuleList
         )) and module.layers[0].__class__.__name__.endswith(suffix):
        for idx, child_module in enumerate(module.layers):
            child_module.register_forward_hook(create_decoderlayer_hook(idx),
                                               with_kwargs=True)

    # Recurse over standard named_children as well, in case nested modules
    # also contain relevant children or their own 'layers' attribute.
    for child_name, child_module in module.named_children():
        register_decoder_layer_PD_disagg_hooks(child_module, suffix)
