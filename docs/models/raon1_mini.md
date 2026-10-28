# Raon-1-mini

Raon-1-mini combines recurrent Raon Delta Attention (RDA), NoPE MLA,
four-stream mHC residuals, and SiTU MoE layers. This implementation reuses
vLLM's native MLA, hybrid cache, quantization, and speculative decoding interfaces.

## Serving

After installing this branch, the intended entry point is:

```bash
vllm serve KRAFTON/Raon-1-mini-preview \
    --reasoning-parser raon_1 \
    --enable-auto-tool-choice --tool-call-parser raon_1
```

Download the published checkpoint from Hugging Face. No model-specific
Python plugin is required.

## Implementation and validation status

This branch is under development. CPU configuration and parser tests do not
qualify model outputs or performance. GPU loading, numerical comparisons,
quality evaluation, and serving benchmarks must pass before public submission.

- RDA applies its separate write/erase gate ratio after the value convolution;
  the rolling convolution history retains unscaled values.
- mHC preserves four residual streams. The model exposes their concatenated
  intermediate states for the native DFlash2 interface.
- MXFP8 experts use vLLM's ModelOpt loader. The checkpoint's E8M0 scale tensor
  suffix `weight_scale_inv` maps to the native `weight_scale` parameter.
  MXFP8 and NVFP4 reuse existing upstream quantization backends. Compatibility
  with SiTU, the backend selected on the target GPU, and numerical and
  performance parity with production must be qualified for each format.
- The `raon_1` parser handles reasoning markers and XML function calls,
  including streamed arguments and request-schema type conversion.
- `--additional-config '{"raon_fast": true}'` enables the ported mHC and RDA
  kernels. This path currently requires SM10x, tensor parallel size 1,
  BF16 convolution history, FP32 recurrent state, and the default SD layout.
  These kernels are awaiting validation against the current upstream runtime.
- NVFP4, FP8 KV cache, prefix caching, and DFlash2 are part of the qualification
  scope. Their availability in upstream vLLM does not establish Raon correctness.
- Pipeline parallelism, expert parallelism, and RecoverSSM are unsupported.

Published checkpoints:

- [MXFP8 target](https://huggingface.co/KRAFTON/Raon-1-mini-preview)
- [NVFP4 target](https://huggingface.co/KRAFTON/Raon-1-mini-preview-nvfp4)
- [DFlash2 draft](https://huggingface.co/KRAFTON/Raon-1-mini-preview-dflash2)
