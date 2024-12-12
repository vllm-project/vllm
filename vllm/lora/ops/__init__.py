from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON and current_platform.is_cuda_alike():
    from vllm.lora.ops.triton_ops.bgmv_expand import bgmv_expand
    from vllm.lora.ops.triton_ops.bgmv_expand_slice import bgmv_expand_slice
    from vllm.lora.ops.triton_ops.bgmv_shrink import bgmv_shrink
    from vllm.lora.ops.triton_ops.sgmv_expand import sgmv_expand
    from vllm.lora.ops.triton_ops.sgmv_expand_slice import sgmv_expand_slice
    from vllm.lora.ops.triton_ops.sgmv_shrink import sgmv_shrink
elif current_platform.is_cpu():
    from vllm.lora.ops.torch_ops.lora_ops import (bgmv_expand,
                                                  bgmv_expand_slice,
                                                  bgmv_shrink, sgmv_expand,
                                                  sgmv_expand_slice,
                                                  sgmv_shrink)

__all__ = [
    "bgmv_expand", "bgmv_expand_slice", "bgmv_shrink", "sgmv_expand",
    "sgmv_expand_slice", "sgmv_shrink"
]
