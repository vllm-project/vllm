# torch.compile with Multimodal Encoders

`torch.compile` can now be applied to multimodal encoders and miscellaneous nn modules in vLLM, including vision-language models like LLaMA 4, Qwen-VL,
and similar encoder-based architectures.

This document covers the basics of how the `torch.compile` integration works for multimodal encoders in vLLM, as well as how to apply the decorator
to new models to improve performance.

!!! note
    For general information about `torch.compile` integration in vLLM, see the [torch.compile design document](./torch_compile.md).

## Overview

We have recently enabled the `@support_torch_compile` decorator to work for multiple nn module components within a model type; this enables
turning compile on for multimodal encoders, bringing performance improvements to additional components of the stack.

When applied to the vision block of [`Qwen2_5_vl`](https://github.com/vllm-project/vllm/pull/23207) we observe ~4.5% e2e perf improvements with
some increase in compilation time

This feature is off by default, but can be enabled by setting `compile_mm_encoder: true` in the compilation config when models have the
`@support_torch_compile` decorator.

## How Compilation Works for Multimodal Components

### APIs for Enablement

To compile a multimodal component such as an encoder, we follow the same mechanism as the LLM text backbone, with a few additional scaffoldings:

1. The `@support_torch_compile` decorator should include `enable_if=should_torch_compile_mm_vit`. This will gate the compilation behind our
`compile_mm_encoder` configuration

2. `with set_model_tag("<component_name>", is_encoder=True)` context manager should be used around the nn.Module's instantiation. Since torch.compile
relies on caching artifacts to reduce start time, we must properly propagate the `<component_name>` information to the cache in order to avoid collisions
with the LLM text-backbone, or other instances of the same artifact (as is the case with vision block). `is_encoder=True` is also needed for encoder
components (see Compile Range Integration).

3. `with set_forward_context` context manager should be used around the nn.Module's forward call. This will properly forward the vllm_config which is needed
for torch.compile integration.

### CompilationConfig

With the exception of `compile_mm_encoder: true`, the multimodal encoder will inherit from the same compilation config as the text LLM. We may extend
this for more configuration in the future.

## Applying torch.compile to a New Multimodal Model/Component

To apply `support_torch_compile` to a new general nn.Module, we advise following the same steps in [`debug_vllm_compile`](./debug_vllm_compile.md); this includes:

1. Applying `support_torch_compile` on initially small modules (such as basic MLP layers), then raising to more general modules until one reaches a good performance
tradeoff

2. Leveraging [`tlparse`](https://github.com/meta-pytorch/tlparse) to identify and eliminate the source of recompiles and graph breaks

3. Using `dynamic_arg_dims` and proper `dynamic_shapes_config` to handle dynamism.

### Common pitfalls

## VllmBackend Feature Support

### Compile ranges

The torch.compile integration will try to rely on max_batch_size to infer compilation ranges for dynamic shapes; however, for modules used in the encoder, this
shape can be difficult to infer due to the unspecified range of shapes the encoder may see as input. Therefore, we rely on `is_encoder=True` in the `set_model_tag`
to alert torch.compile to the fact that this range cannot be inferred, and we default to the range (1, MAX_INT).

!!! note
    We may seek to tighten this range for better performance in the future

### Cudagraphs

vLLM now supports Piecewise CUDA Graph integration for the Vision Transformer (ViT) encoder in Qwen2.5-VL and Qwen3-VL models. This feature captures CUDA graphs at specified patch sizes to reduce kernel launch overhead and improve performance.

#### Enabling ViT CUDA Graphs

**Important**: This feature is **not enabled by default**. The Piecewise CUDA Graph implementation relies on `torch.compile` to trace the computation graph and separate the attention operators. Therefore, users must explicitly enable ViT compilation via the `--compilation-config` argument to activate this feature.

To enable ViT CUDA graph compilation, use:

```bash
vllm serve <model> --compilation-config '{"compile_mm_encoder": true}'
```

#### Configuring Capture Sizes

You can specify custom patch sizes for CUDA graph capture using `vit_cudagraph_capture_sizes`. For models like `Qwen2.5-VL` and `Qwen3-VL`, the capture sizes should be multiples of the square of `merge_size`:

```bash
vllm serve <model> --compilation-config '{"compile_mm_encoder": true, "vit_cudagraph_capture_sizes": [512, 1024]}'
```

Alternatively, you can specify `max_vit_cudagraph_capture_size` to generate a default list of capture sizes up to the given value:

```bash
vllm serve <model> --compilation-config '{"compile_mm_encoder": true, "max_vit_cudagraph_capture_size": 2048}'
```

#### Default Behavior

Once enabled, if `vit_cudagraph_capture_sizes` is not specified, vLLM will use a default set of sizes for capture. Since `compile_mm_encoder` is `False` by default, this feature remains inactive unless configured.

If you only want to enable `torch.compile` for ViT without using the CUDA Graph feature, you can explicitly set the capture sizes to empty:

```bash
vllm serve <model> --compilation-config '{"compile_mm_encoder": true, "vit_cudagraph_capture_sizes": []}'
```

#### Limitations & Notes

- **Image Only**: This feature currently only supports image inference. Video inference is not supported yet.

## Troubleshooting

### Graph Breaks in Vision Encoders

Some vision encoder operations may cause graph breaks. To identify them:

```bash
TORCH_LOGS="+dynamo" vllm serve <MODEL>
```

Common causes of graph breaks in multimodal models:

- **Dynamic image sizes**: Use `dynamic_shapes_config` to handle variable resolutions
- **Untraceable operations**: Some operations (such as to_list) may not be supported by Dynamo
- **Conditional processing**: Data-dependent branching based on image properties

### Compilation Errors

If compilation fails for a multimodal model:

1. **Disable and test**: First verify the model works without compilation:
   ```bash
   VLLM_TORCH_COMPILE_LEVEL=0 vllm serve <model> --compilation-config='{"compile_mm_encoder":"false"}'
   ```

2. **Check logs**: Enable debug logging to see compilation details:
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --compilation-config='{"compile_mm_encoder":"true"}'
   ```

3. **Report issues**: If you find a bug, [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose)

## See Also

- [torch.compile Integration](./torch_compile.md) - Core design document
- [Debugging torch.compile](./debug_vllm_compile.md) - Detailed debugging guide
- [Multimodal Inputs](../features/multimodal_inputs.md) - How to pass multimodal data
- [Disaggregated Encoder](../features/disagg_encoder.md) - Scaling vision encoders
- [Supported Multimodal Models](../models/supported_models.md#list-of-multimodal-language-models) - Model compatibility
