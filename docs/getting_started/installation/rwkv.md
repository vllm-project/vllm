# RWKV-only source build

The opt-in `rwkv` build profile is a reduced CUDA artifact for dense RWKV7 raw
`.pth` checkpoints. It supports the built-in RWKV tokenizer, text
chat/completions, streaming, stop handling, Prometheus metrics, and rapid
sampling with TP=1, PP=1, and DP=1.

Install the explicit dependency set and disable dependency/build isolation when
installing the source tree:

```bash
uv venv --python 3.12 .venv-rwkv
uv pip install --python .venv-rwkv/bin/python -r requirements/rwkv.txt
VLLM_BUILD_PROFILE=rwkv \
VLLM_TARGET_DEVICE=cuda \
uv pip install --python .venv-rwkv/bin/python \
  --no-deps --no-build-isolation --editable .
```

When building a copied source snapshot without `.git` metadata, also set a
deterministic `SETUPTOOLS_SCM_PRETEND_VERSION` (for example, the source
revision's release version). A normal Git checkout does not need this override.

`--no-deps` avoids resolving a second time after the explicit requirements
install. The selected RWKV artifact records that same reduced set in its
package metadata and receives an `rwkv` version label, so dependency-consistency
checks remain meaningful without changing normal full metadata.
`--no-build-isolation` reuses the build tools declared by
`requirements/rwkv.txt` and avoids resolving the full `pyproject.toml` build
environment. The generated `vllm/_build_profile.json` records the profile,
native targets, external projects, architecture, weight format, device,
runner, and TP/PP/DP boundaries that were actually configured.

| Capability | `rwkv` | default `full` |
| --- | --- | --- |
| Dense RWKV7 raw `.pth` | yes | yes |
| Built-in RWKV tokenizer and text OpenAI API | yes | yes |
| Rapid sampler first-use JIT | yes | yes |
| Other architectures or weight formats | no | yes |
| Quantization, multimodal, speculative decoding, LoRA | no | yes |
| Responses, Anthropic, generative-scoring, MCP routes | no | yes |
| Structured-output constraints | no | yes |
| SageMaker container-standard routes | no | yes |
| TP, PP, or DP greater than one; Ray | no | yes |
| Generic stable/MoE operators and attention external projects | no | yes |
| Rust frontend | no | yes |

Unsupported configurations fail during `VllmConfig` validation, before engine
workers start, and request a full build. The profile also rejects precompiled
full extensions rather than relabeling them as reduced artifacts.

To return to normal vLLM compatibility, create a clean environment, install
`requirements/build/cuda.txt` and `requirements/cuda.txt`, and install with
`VLLM_BUILD_PROFILE` unset. Unset and explicit `full` are equivalent.

For clean comparison runs, use:

```bash
tools/rwkv_profile/measure_profile.sh rwkv /path/to/clean/source /path/to/new/output
tools/rwkv_profile/measure_profile.sh full /path/to/other/clean/source /path/to/new/output
```

Each output contains dependency and build logs, resolved distributions,
dependency consistency output, the native target manifest, and size/time
metrics. Use separate source trees, environments, FetchContent roots, and uv
caches for the two profiles.
