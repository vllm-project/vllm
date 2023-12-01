from typing import Optional

import torch


class ExLlamaV2DeviceTensors:

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes
        self.scratch = None

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2, ),
            dtype=torch.half,
            device=f"cuda:{self.device_idx}",
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()
        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice


def quant_post_init(model, max_tokens: Optional[int] = None):
    """
    The max_tokens argument is specific to the exllama backend,
    that requires to initialize a buffer temp_state.
    """
    fixed_bytes = {}

    model_uses_exllama = False
    for _, submodule in model.named_modules():
        if hasattr(submodule, "linear_weights") and getattr(
                submodule.linear_method, "use_exllama", False):
            model_uses_exllama = True
            device = submodule.linear_weights["qweight"].device
            height, width = submodule.linear_weights["qweight"].shape
            scratch_fixed = submodule.linear_method.scratch_space_fixed(
                height * submodule.linear_method.quant_config.pack_factor,
                width, max_tokens)
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
            if hasattr(submodule, "linear_weights") and getattr(
                    submodule.linear_method, "use_exllama", False):
                device = submodule.qweight.device
                submodule.linear_method.post_init(submodule.linear_weights,
                                                  temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model
