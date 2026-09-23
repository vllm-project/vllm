# LiLiCorr

LiLiCorr uses a DFlash backbone to produce per-position candidates, then scores
candidate combinations with a learned correlator. It requires a compatible
trained checkpoint and Model Runner V2, which is selected automatically.
Both plain and grouped-convolution checkpoints are supported.

## Usage

For a checkpoint trained with block size 16:

```bash
vllm serve /path/to/target --dtype bfloat16 \
    --speculative-config '{"method":"dflash","model":"/path/to/lilicorr","num_speculative_tokens":15,"draft_sample_method":"probabilistic"}'
```

Set `num_speculative_tokens` to the checkpoint's trained `block_size - 1`.
The block contains one anchor position and the remaining candidate positions.
The current implementation builds the learned slot embeddings and attention
buffers for that fixed geometry and scores the complete candidate lattice.
Using fewer draft positions would require a separate implementation and quality
validation; it is not supported by changing this setting alone.

## Proposal sampling

- `"draft_sample_method":"greedy"` selects the highest-scoring candidate at each
  step of the correlator's conditional walk.
- `"draft_sample_method":"probabilistic"` samples from those conditional scores
  using the request's temperature and supplies the realized proposal
  distributions to the rejection sampler.

These settings control proposals; they do not replace the target model's
sampling settings. With request temperature zero, probabilistic proposals also
use argmax. For nonzero temperatures, benchmark both modes with your checkpoint
and workload: compare acceptance length and end-to-end throughput or latency.
Neither mode is universally faster.

The default rejection method is `"rejection_sample_method":"standard"`.
The shared Model Runner V2 rejection sampler also exposes `"block"`, but
LiLiCorr-specific GPU correctness and performance validation for that combination
is still pending. Do not assume that changing the rejection method improves
throughput; evaluate it separately from the proposal sampling mode.

## Checkpoint requirements

The checkpoint must declare `LiLiCorrDraftModel` and include all `lilicorr_*`
geometry fields and trained head weights. Geometry is read from `dflash_config`.
Convolution tensors must match the configured `conv_kernel_size` and
`conv_group_size`. Target input embeddings must be available on the draft rank.
By default, candidates use the target LM head, which must also be available there.
A checkpoint with top-level `"has_own_lm_head": true` instead uses its own
`lm_head` with the draft quantization configuration and exclusions. Its weights
must be present; for ModelOpt NVFP4 this includes `lm_head.weight`,
`lm_head.weight_scale`, and `lm_head.weight_scale_2`. The owned head is preserved
and is not replaced by the target head. Candidate token embeddings still come
from the target.

Correlator linear layers and convolution kernel projections use the draft
quantization configuration and its module exclusions. The following parameters
must remain in floating-point model dtype:

- Lattice QKV `in_proj_weight` and `in_proj_bias`, which retain the exported
  parameter layout.
- `lilicorr.factor_input_proj`, `lilicorr.out_head`, and `lilicorr.in_head`, which
  are used by the split factor and fused edge projections. Exclude these modules
  when exporting a quantized checkpoint.
