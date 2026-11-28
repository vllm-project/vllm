# CPU checkpoint loading benchmark

`vllm load-bench` consumes the native `DefaultModelLoader` weight iterator
without constructing a model or initializing a GPU worker. It accepts a local
Hugging Face checkpoint directory. The original file selection, index filtering,
loading strategy and prefetch implementation are reused.

```bash
vllm load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --safetensors-load-strategy lazy

vllm load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --safetensors-load-strategy prefetch \
  --safetensors-prefetch-num-threads 32

vllm load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --safetensors-load-strategy eager

vllm load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'
```

Unspecified settings retain `LoadConfig` defaults. In particular, the serial
iterator may automatically prefetch on recognized network filesystems when the
checkpoint fits in available RAM. Use `lazy` for an explicit no-prefetch baseline.
The native multi-thread loader rejects `prefetch` and `eager` strategies.
`--safetensors-prefetch-block-size` controls each prefetch read in bytes.

Every tensor is copied, in chunks, into one reusable 16 MiB CPU buffer. This
forces access to mmap-backed data without retaining a full model copy. The
original eager and multi-thread iterators can still retain their own tensors
and file buffers. PyTorch's CPU thread count is left unchanged.

The reported JSON measures iterator preparation, iteration and CPU copies.
Throughput is logical tensor payload GiB/s, not storage or network bandwidth.
It excludes model construction, model-specific filtering, TP/EP slicing,
quantization and GPU transfers. Background prefetch is not joined, preserving
the original iterator behavior. Cache state is not changed by this command;
record cold/warm cache conditions separately when comparing runs.

## Prefetch-only benchmark

Use `--prefetch-only` to measure only prefetching. It reuses the loader's file
selection (including safetensors index filtering) and the streaming reader's
file window, interleaved block reads, and per-file I/O executors. It skips
mapping and tensor parsing, joins all workers, and reports the
joined elapsed time and checkpoint-byte throughput. It does not map tensors,
parse safetensors headers into tensors, or copy tensor payloads.

Prefetch-only uses the streaming parameters and defaults to 4 concurrently
active files with 8 IO threads per file. The block size reuses
`--safetensors-prefetch-block-size`. Set them explicitly when comparing runs.

```bash
vllm load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --prefetch-only \
  --prefetch-file-window 4 \
  --io-threads-per-file 8 \
  --safetensors-prefetch-block-size 16777216
```

Unspecified prefetch options retain the `LoadConfig` defaults. Record
cold/warm cache conditions separately; successful prefetch intentionally
populates the page cache and will make the following load benchmark warmer.
Set `VLLM_LOGGING_STREAM=ext://sys.stderr` to keep stdout as JSON when collecting
results programmatically.

An installed vLLM environment and its Python dependencies are still required;
the command bypasses GPU initialization, not package dependencies. For a source
checkout in a compatible environment, the same entrypoint is available through
`python -m vllm.entrypoints.cli.main load-bench ...`.

## Streaming prefetch

The `streaming` reads shard blocks in parallel into the page cache and waits for
all reads of a shard before memory-mapping its tensors. Completed shards are
consumed in completion order while the next shards are being prefetched.

```bash
vllm load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --safetensors-load-strategy streaming
```

The same options work with `vllm serve`. By default, four shards are active
and each uses eight IO threads. The window is replenished as shards
are consumed, so a slow consumer cannot accumulate mappings for the whole
model. Each I/O thread reuses one block-sized buffer. Retained model tensors
and OS page cache are outside this bound; this is not a hard limit on process
RAM or page-cache residency. No cache eviction is performed.
The block size is shared with ordinary prefetch through
`--safetensors-prefetch-block-size`.
When `enable_multithread_load` is set, `streaming` still uses its own file
window and I/O thread settings.

Each rank independently prefetches the files it loads, including on different
nodes. Expert filtering still skips tensor materialization, but prefetch reads
whole shards. Streaming propagates read errors and joins its workers on normal
completion, failure, or iterator close. Unlike ordinary `prefetch`, it does not
leave background prefetch workers running after the benchmark returns.

Compare lazy, prefetch and streaming on the same checkpoint, with separately
recorded cold/warm cache conditions and identical CPU thread settings. No
large-checkpoint or network-storage speedup is implied by the unit tests.

## Portable wheel

Build a pure-Python wheel for this benchmark using the same temporary-staging
approach as SGLang's CPU load wheel:

```bash
# Run from the vllm checkout; use your existing uv environment.
export VIRTUAL_ENV=/Users/sunzhenyuan/.venv
uv pip install --python "$VIRTUAL_ENV/bin/python" build cmake ninja packaging \
  'setuptools>=77.0.3,<81' 'setuptools-scm>=8' 'setuptools-rust>=1.9' \
  'torch==2.14.0' wheel jinja2
"$VIRTUAL_ENV/bin/python" tools/build_cpu_load_wheel.py
```

The default output is `dist/vllm-0.28.1rc0+cpuload1-py3-none-any.whl`.
Use `--version` and `--outdir` to override the version and destination.
The script uses the existing empty-device build, disables Rust compilation,
and excludes prebuilt native libraries and the Rust executable in a temporary
copy. It preserves the Python modules, package data, entrypoint and common
dependencies, and adds the source revision's PyTorch dependency. The original
`setup.py`, `pyproject.toml` and native build remain unchanged.

Install on the benchmark host:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  dist/vllm-0.28.1rc0+cpuload1-py3-none-any.whl
"$VIRTUAL_ENV/bin/vllm" load-bench --model-path /mnt/models/GLM-4.5-Air/ \
  --safetensors-load-strategy lazy
```

This wheel installs as `vllm` and replaces an existing vLLM installation; use a
separate environment from a serving deployment. It is intended for CPU weight
loading, not inference or serving. Its Python dependencies still need wheels
compatible with the target Python and OS. On Linux, a CPU-only PyTorch build can
be installed using `uv pip install --torch-backend=cpu` before installing this
wheel. Run the installed command outside the source checkout to avoid importing
the checkout instead of the wheel.
