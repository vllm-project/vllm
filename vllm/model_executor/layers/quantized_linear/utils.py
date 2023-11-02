from typing import Optional

import torch

from vllm.model_executor.layers.quantized_linear.gptq import (
    GPTQColumnParallelLinear,
    GPTQRowParallelLinear,
    GPTQLinear,
    ExLlamaV2DeviceTensors,
)


def quant_post_init(model, max_tokens: Optional[int] = None):
    """
    The max_tokens argument is specific to the exllama backend,
    that requires to initialize a buffer temp_state.
    """
    fixed_bytes = {}

    model_uses_exllama = False
    for _, submodule in model.named_modules():
        if isinstance(submodule,
                      (GPTQColumnParallelLinear, GPTQRowParallelLinear,
                       GPTQLinear)) and submodule.use_exllama:
            model_uses_exllama = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed(max_tokens)
            fixed_bytes[device] = max(scratch_fixed,
                                      fixed_bytes.get(device, 0))

    if model_uses_exllama:
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(
                device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

        for _, submodule in model.named_modules():
            if isinstance(submodule,
                          (GPTQColumnParallelLinear, GPTQRowParallelLinear,
                           GPTQLinear)) and submodule.use_exllama:
                device = submodule.qweight.device
                submodule.post_init(temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model
