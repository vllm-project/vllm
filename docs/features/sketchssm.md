# SketchSSM

SketchSSM speeds up the decode of Mamba-2 layers in Mamba-2 based models. Each request decodes in windows of `W` steps (`--replayssm-buffer-len`, 16 by default). The last step of a window flushes the window into the exact FP32 state and rebuilds a compact per-request sketch of it; the other steps read the sketch instead of the full state. Dense heads (rank 0) keep BF16 rows of their full state instead, built with the sketch at the flush and after prefill. During decode the FP32 state stays exact and changes only at flushes.

The per-head sketch ranks and orthogonal frames come from an offline calibration of the model ([SketchSSM](https://github.com/SNU-ARC/SketchSSM)).

!!! warning
    SketchSSM is experimental.

## Usage

```bash
vllm serve <model> --sketchssm /path/to/calibration.pt
```

`--sketchssm` takes a file, a directory, or a Hugging Face repository id that contains `calibration.pt`. The file holds either:

- **Exported frames for one rank budget:** `frames[L, G, K, K]`, `m_table[L, H]` (sketch rank per head, 0 = dense), and optionally `dense_table[L, H]` and `layer_ids`.
- **A portable calibration:** the per-group basis and rank-score curves. `--sketchssm-mean-rank` sets the mean sketch rank per head (default 8); ranks and frames are allocated at load time for the serving window.

## Constraints

SketchSSM currently supports NemotronH (`NemotronHForCausalLM`) models. It requires a CUDA GPU with compute capability 8.0 or newer, Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`), `--mamba-backend triton` and `--mamba-ssm-cache-dtype float32`. It replaces `--use-replayssm`; do not pass both.

Mamba prefix caching works in align mode (`--mamba-cache-mode align`), as with the default decode. Tensor parallelism splits the heads, their ranks and the group frames across ranks (the tensor-parallel size must divide the heads, and the groups unless there is a single group). Speculative decoding, stochastic rounding of the SSM state and KV connectors are not supported yet.

Mamba-2 layers need a power-of-two state size of at least 16 and a head dim of at least 16.
