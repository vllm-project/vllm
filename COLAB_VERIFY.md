# Independent Colab verification

Run the GPU check on a Colab CUDA runtime with a different GPU from the local
RTX 3080 run. The branch in this workspace is not published yet; once it is
shared, replace `PR_SHA` below with its commit and `BASE_SHA` with the merge
base against the target branch.

Set these to the candidate commit and target-branch merge base:

```bash
export PR_SHA=REPLACE_WITH_CANDIDATE_SHA
export BASE_SHA=REPLACE_WITH_MERGE_BASE_SHA
```

## CUDA GPU: correctness and end-to-end comparison

1. In Colab, select a GPU runtime (L4, A100, or another supported NVIDIA GPU).
   Capture `nvidia-smi`, `python --version`, and `torch.__version__` in the
   result report. Use the same runtime for baseline and candidate runs.
2. Clone vLLM, check out the candidate commit, and install the checkout using
   the vLLM [CUDA source install steps](docs/getting_started/installation/gpu.cuda.inc.md).
   This change is Python/Triton JIT code. A precompiled install is also fine
   when a wheel for the exact checkout is available; avoid mixing source from
   this branch with an unrelated vLLM wheel.

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git fetch origin "$PR_SHA"
   git checkout --detach "$PR_SHA"
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e . --torch-backend=auto
   uv pip install pytest
   ```

3. Run the mask topology test matrix on the candidate:

   ```bash
   pytest -q --confcutdir=tests/kernels \
     tests/kernels/test_flex_attention.py \
     -k 'encoder_span or default_block_sizes'
   ```

   It checks exact token-level visibility for irregular and aligned packed
   requests, many one-token requests, four query/KV block-size pairs, plus the
   row sentinels consumed by FlexAttention. The
   expected result is all selected tests passing; a skip means Triton or CUDA
   was unavailable and does not verify the GPU path.

4. Save the benchmark script outside the checkout so it remains available
   while switching revisions, then run the same script on baseline and
   candidate in fresh Python processes. Use `BASE_SHA` from the PR merge base.

   ```bash
   cp benchmarks/kernels/benchmark_flex_attention_encoder.py \
     /content/benchmark_flex_attention_encoder.py
   git checkout --detach "$BASE_SHA"
   python /content/benchmark_flex_attention_encoder.py \
     --profile pack4x2048 --q-block-size 128 --kv-block-size 128 \
     --output /content/base-pack4.json

   git checkout --detach "$PR_SHA"
   python /content/benchmark_flex_attention_encoder.py \
     --profile pack4x2048 --q-block-size 64 --kv-block-size 64 \
     --reference /content/base-pack4.npz \
     --output /content/pr-pack4.json
   ```

   Repeat with `--profile irregular8192` and save separate output files. For
   latency, do three paired baseline/candidate runs, alternating which revision
   runs first. Each run uses four warmups and 15 timed repetitions by default.
   Report the three p50 pairs and their geometric-mean speedup, along with the
   minimum embedding cosine and max absolute difference. The script fails if
   minimum cosine is below `0.999`; investigate any failure instead of
   relaxing the threshold. Keep the exact JSON and NPZ files as artifacts.

   Check out each revision only between benchmark processes. The source is
   installed editable, so a fresh process imports that revision. If the
   environment uses a non-editable install, reinstall vLLM at each checkout.

## TPU runtime: fallback boundary only

The new builder is deliberately limited to CUDA-alike platforms with Triton
and PyTorch 2.9 or newer. TPU is expected to retain the generic builder and
128-token default. This PR makes no TPU performance claim.

If the Colab TPU runtime has the vLLM TPU hardware plugin required by the
current vLLM checkout, first run a normal supported model smoke test with
`FLEX_ATTENTION` and confirm the model loads. Then verify the platform gate and
fallback block-size selection:

```bash
python - <<'PY'
from vllm.config import AttentionConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadataBuilder

assert not current_platform.is_cuda_alike()
assert FlexAttentionMetadataBuilder._get_block_sizes(
    AttentionConfig(),
    supports_small_blocks=False,
    cache_block_size=16,
    uses_paged_kv=False,
    supports_small_encoder_blocks=False,
) == (128, 128)
print("TPU remains on the generic 128x128 encoder path")
PY
```

If the TPU plugin does not expose `FLEX_ATTENTION` or the ModernBERT model,
record the plugin/runtime error and report the static gate check only. Do not
count this as runtime validation of the GPU kernel or as a TPU speedup result.
