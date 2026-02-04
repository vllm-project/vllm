# CustomOp

`CustomOp` is an abstract class used for dispatching the forward method of various operations to the appropriate backend. It also offers a mechanism for both vLLM and OOT (Out-Of-Tree) plugins to register their custom operations.

This document will introduce how CustomOp works in vLLM and how to implement a new `CustomOp`.

## How CustomOp Works in vLLM

`CustomOp` manages two dictionaries of all custom ops (i.e., op classes, indexed by registered name) in its class, for vLLM and OOT plugins respectively.

We can use `@CustomOp.register("op_name")` to register an op class to the `CustomOp` system. After this, the `op_name` and its class will be added into the `op_registry` dictionary. In addition, We can also register an OOT op by `@CustomOp.register_oot("op_name")`. We will introduce this mechanism in detail later.

When a `CustomOp` is called (i.e., call its `forward()` method), if it is enabled (i.e., with `--compilation_config.custom_ops '["+op_name"]'`), it will automatically dispatch the forward method to the appropriate backend according to `current_platform`. Otherwise (i.e., it is disabled), it will only call the `forward_native()` method to use PyTorch-native implementation of this forward method.

- **CPU platform:** dispatch to `forward_cpu()`.
- **CUDA platform:** dispatch to `forward_cuda()`.
- **ROCm platform:** dispatch to `forward_hip()`. If `forward_hip()` is not implemented, it will use `forward_cuda()` as a fallback.
- **XPU platform:** dispatch to `forward_xpu()`.
- **TPU platform:** dispatch to `forward_tpu()`.
- **OOT platform:** dispatch to `forward_oot()`. This will only be called on OOT platforms.
- **Default:** dispatch to `forward_native()` as a final fallback for all platforms.

!!! note
    Note that the dispatching logic might not be absolute because of class inheritance. Derived class might override the behavior.

Furthermore, vLLM decides whether to enable or disable a `CustomOp` based on `compilation_config.custom_ops`. To be specific, if a `CustomOp` is not registered in `compilation_config.custom_ops` (i.e., uses the default config), it will be enabled if `compilation_config.custom_ops` contains `all`, or will be disabled if it contains `none`.

!!! note
    Note that `all` and `none` cannot coexist in `compilation_config.custom_ops`.

By default, if `compilation_config.backend == "inductor"` and `compilation_config.mode != CompilationMode.NONE`, a `none` will be appended into `compilation_config.custom_ops`, otherwise a `all` will be appended. In other words, this means `CustomOp` will be disabled in some platforms (i.e., those use `inductor` as default backend for `torch.compile`) when running with torch compile mode. In this case, Inductor generates (fused) Triton kernels for those disabled custom ops.

!!! note
    For multi-modal models, vLLM has enforced the enabling of some custom ops to use device-specific deep-optimized kernels for better performance in ViT part, such as `MMEncoderAttention` and `ApplyRotaryEmb`. We can also pass a `enforce_enable=True` param to the `__init__()` method of the `CustomOp` to enforce enable itself at object-level.

    Note that this `enforce_enable` mechanism will be removed after we add a separate `compilation_config` for multi-modal part.

## How to Customise Your Configuration for CustomOp

vLLM also offers fine-grained control over which custom ops to enable or disable for users, by manually passing a `--compilation_config.custom_ops '["..."]'` when launching a server.

For example:

- Use `--compilation_config.custom_ops '["all"]'` to enable all custom ops.
- Use `--compilation_config.custom_ops '["none"]'` to disable all custom ops.
- Use `--compilation_config.custom_ops '["all,-op1"]'` to enable all custom ops except op1 (i.e., prefixed with a `-` means "disable").
- Use `--compilation_config.custom_ops '["none,+op1,+op2"]'` to only enable op1 and op2 (i.e., prefixed with a `+` means "enable").

## Types of Supported CustomOp in vLLM

**1. Attention:**

```python
--8<-- "vllm/model_executor/layers/attention/mm_encoder_attention.py:mm_encoder_attn"

--8<-- "vllm/model_executor/layers/attention/rel_pos_attention.py:rel_pos_attention"

--8<-- "vllm/model_executor/layers/mla.py:multi_head_latent_attention"
```

**2. Activation:**

```python
--8<-- "vllm/model_executor/layers/activation.py:silu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:mul_and_silu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_new"

--8<-- "vllm/model_executor/layers/activation.py:gelu_fast"

--8<-- "vllm/model_executor/layers/activation.py:quick_gelu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul_sparse"

--8<-- "vllm/model_executor/layers/activation.py:relu2"

--8<-- "vllm/model_executor/layers/activation.py:xielu"

--8<-- "vllm/model_executor/layers/activation.py:swigluoai_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:fatrelu_and_mul"
```

**3. MM-Conv:**

```python
--8<-- "vllm/model_executor/layers/conv.py:conv2d"

--8<-- "vllm/model_executor/layers/conv.py:conv3d"
```

**4. Embedding:**

```python
--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:vocab_parallel_embedding"

--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:parallel_lm_head"
```

**5. Linear:**

```python
--8<-- "vllm/model_executor/layers/linear.py:row_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:column_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:replicated_linear"
```

**6. Logits Processor:**

```python
--8<-- "vllm/model_executor/layers/logits_processor.py:logits_processor"
```

**7. Mamba:**

```python
--8<-- "vllm/model_executor/layers/mamba/mamba_mixer.py:mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mamba_mixer2"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mixer2_gated_rms_norm"

--8<-- "vllm/model_executor/models/plamo2.py:plamo2_mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/short_conv.py:short_conv"
```

**8. MoE:**

```python
--8<-- "vllm/model_executor/layers/fused_moe/layer.py:fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py:modular_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:unquantized_fused_moe"

--8<-- "vllm/model_executor/models/transformers/moe.py:transformers_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/fused_moe.py:grouped_topk"
```

**9. Norm:**

```python
--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm"

--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm_gated"

--8<-- "vllm/model_executor/layers/layernorm.py:gemma_rms_norm"
```

**10. Quantization:**

```python
--8<-- "vllm/model_executor/layers/quantization/input_quant_fp8.py:quant_fp8"
```

**11. Rope:**

```python
--8<-- "vllm/model_executor/layers/rotary_embedding/base.py:rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py:dual_chunk_rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/common.py:apply_rotary_emb"
```

## Guidelines for Implementing a New CustomOp

### Implement a New CustomOp in vLLM

This part is a tutorial of how to implement a New `CustomOp` in vLLM.

Steps:

1. Implement a new op class, which extends from `CustomOp` base class.
2. Add the `@CustomOp.register("op_name")` decorator on this op class to register it into `CustomOp` system.
3. Implement different `forward_xxx()` method according to your needs.

Taking `MMEncoderAttention` as an example:

??? code

    ```python
    @CustomOp.register("mm_encoder_attn")
    class MMEncoderAttention(CustomOp):

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
            multimodal_config: MultiModalConfig | None = None,
        ) -> None:
            super().__init__()
            # Init...

        def forward_native(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call TORCH_SDPA implementation...

        def forward_cuda(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call FA or TORCH_SDPA implementation...

        def forward_cpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call TORCH_SDPA implementation...

        def forward_xpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call FA implementation...

        def forward_tpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call PALLAS implementation...
    ```

### Register a New CustomOp in OOT Device Plugins

Currently, thanks to [vLLM's hardware-plugin mechanism](./plugin_system.md), there are various OOT device plugins emerging out to enable vLLM seamlessly runs on different hardwares. You can also find more details about this mechanism at [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html).

- **Official device plugins:** [vllm-ascend](https://github.com/vllm-project/vllm-ascend) (for Huawei Ascend NPU), [vllm-spyre](https://github.com/vllm-project/vllm-spyre)
(for Spyre), [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) (for Intel Gaudi), [vllm-neuron](https://github.com/vllm-project/vllm-neuron) (for AWS Neuron), [vllm-meta](https://github.com/vllm-project/vllm-metal) (for Apple Silicon), etc.
- **Non-official device plugins:** [vllm-metax](https://github.com/MetaX-MACA/vLLM-metax) (for MetaX GPU), [vllm-kunlun](https://github.com/baidu/vLLM-Kunlun) (for Baidu Kunlun XPU), etc.

In this case, `CustomOp` can enable these hardware manufacturers to seamlessly replace vLLM's operations with their deep-optimized kernels for specific devices at runtime, by just registering an OOT `CustomOp` and implementing the `forward_oot()` method.

Now, this part will show you how to register an OOT `CustomOp` for a device plugin.

Taking `MMEncoderAttention` as an example:

1. Implement a `CustomMMEncoderAttention` class which extends from `MMEncoderAttention` and implement its `forward_oot()` method.
2. Register your `CustomMMEncoderAttention` into vLLM to replace `MMEncoderAttention`.

??? code

    ```python
    from vllm.model_executor.layers.attention import MMEncoderAttention
    from vllm.model_executor.custom_op import CustomOp


    @CustomOp.register_oot("MMEncoderAttention")
    class CustomMMEncoderAttention(MMEncoderAttention):

        def __init__(...):
            super().__init__(...)
        
        def forward_oot(...):
            # Call optimized device-specific kernels.
            ...
    ```

In this case, a new item `{"MMEncoderAttention": CustomMMEncoderAttention}` will be added into `op_registry_oot`. When initializing a `MMEncoderAttention` op object, if the class name (i.e., `MMEncoderAttention`) is contained in the keys of `op_registry_oot`, vLLM will replace it with our registered class (i.e., `CustomMMEncoderAttention`) and instantiate it.

After that, when this `MMEncoderAttention` op is called, your `forward_oot()` will be called if it is enabled. Thus, you will get expected performance on your hardwares without directly modify vLLM.

In addition, you can also register all your `CustomOp` at one place for better management.

??? code

    ```python
    from vllm.model_executor.custom_op import CustomOp


    REGISTERED_CUSTOM_OPS = {
        "CustomOP1": YourCustomOp1,
        "CustomOP2": YourCustomOp2,
        "CustomOP3": YourCustomOp3,
    }

    for op_name, op_cls in REGISTERED_CUSTOM_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=op_name)
    ```
