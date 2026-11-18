# Experimental ROCm AITER FA attention paths

These opt-in paths use `ROCM_AITER_FA`. They are intended for qualified ROCm/AITER deployments; they do not change the default attention path.

## Multi-token verification with paged attention

`VLLM_ROCM_AITER_FA_PAGED_MTP` is read at process startup:

| Value | Eligible uniform causal multi-token decode batches |
| --- | --- |
| `0` (default) | Keep the existing unified-attention fallback |
| `full` | Use paged MTP only in full-attention layers |
| `1` | Also allow supported sliding-window layers |

Select the backend with `--attention-backend ROCM_AITER_FA`. The builder checks uniform verification length; unsupported batches keep the existing path. Sliding-window metadata includes the keys required by the earliest query in the verification block. This setting is for target verification, not a switch to a different draft backend.

!!! warning
    This draft integration requires an AITER `paged_attention_v1` launcher that accounts for the MTP factor in its `exp_sums`, `max_logits` and `tmp_out` workspace regions. The historical benchmark used a patched launcher; the exact supported public revision is still being qualified. Leave the setting at `0` on an unqualified AITER build: the flag does not repair an incompatible launcher.

Measure complete decode steps and verify all query rows, page/window boundaries and graph padding before enabling the path. The previous implementation already batches verification queries; this option does not turn eight separate API calls into one.
