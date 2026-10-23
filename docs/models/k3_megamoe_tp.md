# Experimental Kimi-K3 TP MegaMoE on ROCm

This opt-in path targets Kimi-K3 MXFP4 routed experts on eight gfx950 GPUs.
It uses FP8 activations, SiTUv2 (beta 4, linear beta 25), and the existing router.
It requires an AITER build with MegaMoeTP and explicit FP8 SiTUv2 fused-MoE support.

Select it with `--enforce-eager --additional-config '{"k3_megamoe_tp":"mega"}'`.
The default is `off`. `split` selects the same A8W4 split-MoE implementation
used for large prefill, for a precision-matched comparison. The existing
default MXFP4 backend may use a different activation precision.

Supported deployment: TP8, PP1, DP1, BF16 model tensors, no EP, context/sequence
parallelism, ubatching, DBO, speculative decoding, LoRA, or specialized decode.
Unsupported configurations raise before expert execution. Shared experts execute
serially on the main stream; the persistent kernel must have exclusive GPU use.

The model owns one communication runtime shared across its MoE layers. Weight
loading converts the original separated MXFP4 checkpoint bytes once and releases
the original raw storage. MegaMoE stores standard separated packing; split mode
stores the FP8 split kernels' interleaved packing. Global token counts up to 256
use MegaMoE, padded to a multiple of eight. Larger counts temporarily unpack and
repack the same quantized bytes for split A8W4 plus TP all-reduce. This conversion
adds prefill work and transient memory, without retaining both formats per layer.
Whole-model graph execution is currently rejected: default compilation and
no-Inductor full-decode graph both failed GSM8K, despite passing isolated-layer
graph tests. These operator tests do not establish model graph correctness.
A reduced latent output is normalized without another reduction; its
up-projection shard is merged with the shared partial before the final reduction.

This is an experimental integration. Random operator validation alone does not
establish model accuracy, quality, performance, or deployment readiness. Validate
real checkpoint loading, prefill/decode boundaries, graphs, and model evaluation
against the split baseline before using it for a workload. Weight reload is
unsupported; construct a new model instance instead.
