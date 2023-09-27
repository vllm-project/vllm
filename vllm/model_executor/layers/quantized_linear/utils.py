from typing import Optional

import torch

from vllm import quantization_ops
from vllm.model_executor.layers.quantized_linear.gptq import (
    GPTQColumnParallelLinear, GPTQRowParallelLinear, GPTQLinear)


def quant_post_init(model, max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend,
    that requires to initialize a buffer temp_state.
    """
    device_to_buffers_size = {}

    model_uses_exllama = False
    use_act_order = False
    for _, submodule in model.named_modules():
        if isinstance(submodule,
                      (GPTQColumnParallelLinear, GPTQRowParallelLinear,
                       GPTQLinear)) and submodule.use_exllama:
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1
                }

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(
                device_to_buffers_size[device]["max_dq_buffer_size"],
                submodule.qweight.numel() * 8)

            in_features = submodule.input_size_per_partition if isinstance(
                submodule, GPTQRowParallelLinear) else submodule.input_size
            out_features = submodule.output_size_per_partition if isinstance(
                submodule, GPTQColumnParallelLinear) else submodule.output_size
            if submodule.quant_config.desc_act:
                use_act_order = True
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(
                    device_to_buffers_size[device]["max_inner_outer_dim"],
                    in_features, out_features)

    if model_uses_exllama:
        device_to_buffers = {}
        max_input_len = max_input_length if use_act_order else 1
        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order
            # case. The temp_dq buffer is required to dequantize weights when
            # using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state":
                torch.zeros(
                    (max_input_len, buffers_size["max_inner_outer_dim"]),
                    dtype=torch.float16,
                    device=device),
                "temp_dq":
                torch.zeros((1, buffers_size["max_dq_buffer_size"]),
                            dtype=torch.float16,
                            device=device),
                "max_dq_buffer_size":
                buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim":
                buffers_size["max_inner_outer_dim"],
            }

        # Buffers need to be persistent to avoid any bug.
        model.device_to_buffers = device_to_buffers

        for device, buffers in model.device_to_buffers.items():
            quantization_ops.gptq_prepare_buffers(device,
                                                  buffers["temp_state"],
                                                  buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        quantization_ops.gptq_set_tuning_params(matmul_recons_thd,
                                                matmul_fused_remap,
                                                matmul_no_half2)

        # The buffers need to have been initialized first before calling
        # make_q4.
        for _, submodule in model.named_modules():
            if isinstance(
                    submodule,
                (GPTQColumnParallelLinear, GPTQRowParallelLinear, GPTQLinear)):
                submodule.post_init()

        torch.cuda.empty_cache()

    return model
