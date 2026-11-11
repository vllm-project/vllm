# HALOps Linear Backend

[HALOps](https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/) provides
high-performance MXFP8 W8A8 GEMM kernels for NVIDIA Blackwell (SM100) GPUs. It is
published on the PaddlePaddle package index instead of PyPI, so it is an opt-in
extra and is never part of vLLM's default install.

## Installation

```bash
uv pip install "vllm[halops]" \
    -f https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/
```

The same pin is available as a standalone requirements file:

```bash
pip install -r requirements/halops.txt
```

`--find-links` adds a wheel location rather than an index, and the page hosts
only `halops` wheels, so resolution for every other package is unchanged. The
published wheels target CUDA 13.2 (`cu132`); use the `-f` URL that matches the
CUDA build of your `torch`.

## Usage

Select the backend explicitly:

```bash
vllm serve <model> --linear-backend halops
```

`halops` only implements MXFP8 linear layers. Other quantization schemes and
layer types fall back to automatic selection. Automatic selection treats HALOps
as the last MXFP8 candidate, after the established backends, so pass
`--linear-backend halops` to actually use it.

## Supported Configurations

| Backend | Supported configurations |
| ------- | ------------------------ |
| Linear | MXFP8 W8A8: `fp8_e4m3` weights and activations with per-1x32 `e8m0` block scales, `K` and `N` divisible by 128 |

The backend requires a Blackwell (SM100) device.

If the `halops` package is not importable, `--linear-backend halops` fails with
an actionable error, and automatic selection skips the backend.
