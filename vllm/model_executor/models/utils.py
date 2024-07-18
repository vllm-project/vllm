from typing import Callable, Dict, List, Tuple

import torch
from torch.func import functional_call

from vllm.multimodal import BatchedTensors
from vllm.utils import is_pin_memory_available


def merge_vision_embeddings(input_ids: torch.Tensor,
                            inputs_embeds: torch.Tensor,
                            vision_embeddings: BatchedTensors,
                            image_token_id: int) -> torch.Tensor:
    """
    Merge `vision_embeddings` into `inputs_embeds` by overwriting the positions
    in `inputs_embeds` corresponding to placeholder image tokens in `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    mask = (input_ids == image_token_id)
    num_expected_tokens = mask.sum()

    if isinstance(vision_embeddings, torch.Tensor):
        batch_size, batch_tokens, *_, embed_dim = vision_embeddings.shape
        total_tokens = batch_size * batch_tokens
        if num_expected_tokens != total_tokens:
            expr = f"{batch_size} x {batch_tokens}"
            raise ValueError(
                f"Attempted to assign {expr} = {total_tokens} "
                f"image tokens to {num_expected_tokens} placeholders")

        inputs_embeds[mask] = vision_embeddings.view(total_tokens, embed_dim)
    else:
        size_per_batch = [t.shape[0] for t in vision_embeddings]
        total_tokens = sum(size_per_batch)
        if num_expected_tokens != total_tokens:
            expr = ' + '.join(map(str, size_per_batch))
            raise ValueError(
                f"Attempted to assign {expr} = {total_tokens} "
                f"image tokens to {num_expected_tokens} placeholders")

        inputs_embeds[mask] = torch.cat(vision_embeddings)

    return inputs_embeds


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes


def maybe_offload_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    device = next(module.parameters()).device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break

        # `torch.empty_like` does not support `pin_memory` argument
        cpu_data = torch.empty(size=p.data.size(),
                               dtype=p.data.dtype,
                               layout=p.data.layout,
                               device='cpu',
                               pin_memory=pin_memory)
        cpu_data.copy_(p.data)
        p.data = cpu_data
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()

    state_dict: Dict[str, torch.Tensor] = module.state_dict()

    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        device_state = {
            # here we blindly call `to(device)`
            # if the parameter is already on the device, it will be a no-op
            k: v.to(device, non_blocking=True)
            for k, v in state_dict.items()
        }
        output = functional_call(module,
                                 device_state,
                                 args=args,
                                 kwargs=kwargs)
        module.forward = forward
        return output

    module.forward = forward

    return module


def make_layers(
    num_hidden_layers: int, layer_fn: Callable[[], torch.nn.Module]
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn())
            for _ in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: Dict[int, List[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> List[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + '.')
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    for missing_layer_name in get_pp_missing_layer_names(model):
        if name.startswith(missing_layer_name):
            return True
    return False
