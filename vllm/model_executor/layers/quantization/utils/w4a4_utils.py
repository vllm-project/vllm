class Fp4LinearOp:
    """
    This class executes a MX-FP4 linear layer using QDQ.
    """

    def __init__(self,
                 use_per_token_if_dynamic: bool = False):
        self.cutlass_fp8_supported = cutlass_fp8_supported
        self.use_per_token_if_dynamic = use_per_token_if_dynamic

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        input_scale: Optional[torch.Tensor] = None,
        input_scale_ub: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        # TODO(luka) remove this parameter in favor of __init__
        use_per_token_if_dynamic: Optional[bool] = None
    ) -> torch.Tensor:
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.

        # View input as 2D matrix for fp8 methods
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]

        # TODO(luka) this is here because currently MLA only decides this
        #  during the forward method instead of in __init__.
        if use_per_token_if_dynamic is None:
            use_per_token_if_dynamic = self.use_per_token_if_dynamic

        if out_dtype is None:
            out_dtype = input.dtype

        # torch.scaled_mm supports per tensor weights + activations only
        # so fallback to naive if per channel or per token
        if input.dtype != current_platform.fp8_dtype():
            # Maybe apply padding to output, see comment in __init__
            qinput, x_scale = ops.scaled_fp8_quant(
                input_2d,
                input_scale,
                num_token_padding=self.output_padding,
                use_per_token_if_dynamic=use_per_token_if_dynamic)
        else:
            qinput, x_scale = input_2d, input_scale

        # TODO: implement
        
        return output.to(dtype=input.dtype).view(*output_shape)

