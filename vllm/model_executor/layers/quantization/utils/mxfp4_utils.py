import torch

OCP_MX_BLOCK_SIZE = 32

def per_token_group_quant_mxfp4(x: torch.Tensor, block_k: int):
    try:
        from quark.torch.quantization.utils import even_round
        from quark.torch.kernel import scaled_fake_quantize
        from quark.torch.quantization.utils import reshape_to_blocks
    except ImportError as e:
        raise ImportError(f"The package `amd-quark` is required to use "
        "MX-FP4 models. Please install it with `pip install "
        "amd-quark`. Error: {e}")
    
    axis = -1
    block_x = reshape_to_blocks(x, block_k, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    amax = amax.squeeze(-1)

    scale = even_round(amax, "fp4")

    x_qdq = scaled_fake_quantize(  # type: ignore[attr-defined]
        "fp4",
        x,
        scale.to(x.device),
        None,
        axis,
        block_k,
        None,
        None,
        None,
        "per_group",
        None,
    )

    return x_qdq