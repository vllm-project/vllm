# Kerutils

Kerutils is a library that provides:

- Wrappers for low-level PTX instructions
- Helpers for checking tensor device / shape / stride / dtype, useful in kernel library's dispatcher

> This copy is a subset of upstream Kerutils: the CuTe UTCMMA / 2SM TMA copy wrappers,
> the GeMM helpers and the Kernel Insight (KI) library have been removed because
> DeepSelect does not use them.

## Getting Started

To use Kerutils, simply clone this repo (or register it as a submodule, if you're working in a repo), and include `<kerutils/kerutils.cuh>`. You should add `include/` into your compiler's `includePath` (for example, by adding `-Ipath/to/kerutils/include`).

## Organization

- `include/host` - Host-side helper functions
- `include/device` - Device-side PTX wrappers and helper functions, organized by GPU architectures
- `supplemental/` - Files in this directory are NOT included by `kerutils/kerutils.cuh` by default, to reduce compilation time. `supplemental/torch_tensors.h` provides various handy guard functions for examining a libTorch tensor.
