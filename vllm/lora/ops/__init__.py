from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton.bgmv_expand import bgmv_expand
    from vllm.lora.ops.triton.bgmv_expand_slice import bgmv_expand_slice
    from vllm.lora.ops.triton.bgmv_shrink import bgmv_shrink
    from vllm.lora.ops.triton.sgmv_expand import sgmv_expand
    from vllm.lora.ops.triton.sgmv_expand_slice import sgmv_expand_slice
    from vllm.lora.ops.triton.sgmv_shrink import sgmv_shrink
else:
    from vllm.lora.ops.default.lora_ops import (bgmv_expand, bgmv_expand_slice,
                                                bgmv_shrink, sgmv_expand,
                                                sgmv_expand_slice, sgmv_shrink)

__all__ = [
    "bgmv_expand", "bgmv_expand_slice", "bgmv_shrink", "sgmv_expand",
    "sgmv_expand_slice", "sgmv_shrink"
]
