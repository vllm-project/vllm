# Share KV cache from SGLang to vLLM

This example computes a prompt's KV cache in SGLang, stores it in one LMCache
multiprocess daemon, and reuses it in a separate vLLM instance. It verifies
reuse from the LMCache read counter and checks that both engines tokenize the
prompt identically. A cache-isolated cold vLLM request provides the correctness
reference for the later cache-hit vLLM request.

## Requirements

- Linux with two NVIDIA GPUs with at least 80 GB of memory each. The validation
  used two 96 GB GPUs.
- Python 3.12 and `uv`.
- Enough CPU memory for an 8 GB LMCache L1 pool.

The following were the latest releases when this example was validated on
2026-07-17. They are pinned so a later release cannot silently change the test:

- LMCache `0.5.1`
- SGLang `0.5.15.post1`
- vLLM `0.25.1`
- `Qwen/Qwen2.5-32B-Instruct` revision
  `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`

## Install

Create separate environments so each serving engine resolves only its own
runtime dependencies:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_LINK_MODE=copy

uv venv --python 3.12 .venv-sglang
uv pip install --prerelease=allow --python .venv-sglang/bin/python \
  "lmcache==0.5.1" "sglang==0.5.15.post1"

uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python \
  "lmcache==0.5.1" "vllm==0.25.1"
```

SGLang `0.5.15.post1` pins a prerelease FlashAttention wheel, which is why its
install command explicitly uses `--prerelease=allow`. The dedicated uv cache
keeps this installation isolated from shared cache entries, while copy mode
supports hosts where the uv cache and virtual environments are on different
filesystems.

## Run

From this directory:

```bash
SGLANG_VENV=$PWD/.venv-sglang \
VLLM_VENV=$PWD/.venv-vllm \
./verify.sh
```

The script starts one LMCache daemon and both engines. It first sends a salted
vLLM request as a guaranteed cache miss, then sends the unsalted prompt to
SGLang and vLLM. It exits with `PASS` only when:

1. SGLang and both vLLM requests report the same prompt-token count;
2. the LMCache daemon reports one or more L1 chunk reads during the first vLLM
   request after SGLang populates the cache; and
3. greedy vLLM decoding produces identical text for the cache-isolated cold
   request and the cross-engine cache-hit request.

The SGLang process sets `FLASHINFER_USE_CUDA_NORM=1` so FlashInfer `0.6.12`
uses its CUDA JIT normalization fallback instead of the CuTeDSL path on the
validated Blackwell host.

The fixed model revision and identical model name, tensor-parallel size, KV
dtype, page size, and LMCache chunk size are part of the cache identity and
layout contract. Do not change only one engine's values.

## Clean-environment validation

The commands above were tested twice on 2026-07-17 with
`Qwen/Qwen2.5-32B-Instruct` on two NVIDIA RTX PRO 6000 Blackwell Server Edition
GPUs. Before the second run, both virtual environments and the run-specific uv
package cache were deleted, then recreated from empty directories. This checks
that the example does not depend on an editable checkout or packages left in an
old environment.

Both runs produced the same result:

| Run | Prompt tokens | SGLang tokens stored | vLLM L1 read chunks | Correctness |
| --- | ---: | ---: | ---: | --- |
| Initial clean install | 3,688 | 3,584 | 14 | cold and cache-hit vLLM text matched |
| Recreated environments | 3,688 | 3,584 | 14 | cold and cache-hit vLLM text matched |

The salted cold reference read zero L1 chunks in both runs. The cache-hit vLLM
request generated ` The main idea is that LMCache enables multiple independent
inference engines to share and reuse` in both runs. LMCache stores and retrieves
only complete 256-token chunks, so 3,584 of the 3,688 prompt tokens were shared.
