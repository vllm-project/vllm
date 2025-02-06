# GPU

vLLM is a Python library that supports the following GPU variants. Select your GPU type to see vendor specific instructions:

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:selected:
:sync: cuda

:::{include} cuda.inc.md
:start-after: "# Installation"
:end-before: "## Requirements"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "# Installation"
:end-before: "## Requirements"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "# Installation"
:end-before: "## Requirements"
:::

::::

:::::

## Requirements

- OS: Linux
- Python: 3.9 -- 3.12

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "## Requirements"
:end-before: "## Set up using Python"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "## Requirements"
:end-before: "## Set up using Python"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "## Requirements"
:end-before: "## Set up using Python"
:::

::::

:::::

## Set up using Python

### Create a new Python environment

:::{include} ../python_env_setup.inc.md
:::

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "## Create a new Python environment"
:end-before: "### Pre-built wheels"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

There is no extra information on creating a new Python environment for this device.

::::

::::{tab-item} Intel XPU
:sync: xpu

There is no extra information on creating a new Python environment for this device.

::::

:::::

### Pre-built wheels

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "### Pre-built wheels"
:end-before: "### Build wheel from source"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "### Pre-built wheels"
:end-before: "### Build wheel from source"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "### Pre-built wheels"
:end-before: "### Build wheel from source"
:::

::::

:::::

(build-from-source)=

### Build wheel from source

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "### Build wheel from source"
:end-before: "## Set up using Docker"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "### Build wheel from source"
:end-before: "## Set up using Docker"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "### Build wheel from source"
:end-before: "## Set up using Docker"
:::

::::

:::::

## Set up using Docker

### Pre-built images

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "### Pre-built images"
:end-before: "### Build image from source"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "### Pre-built images"
:end-before: "### Build image from source"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "### Pre-built images"
:end-before: "### Build image from source"
:::

::::

:::::

### Build image from source

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "### Build image from source"
:end-before: "## Supported features"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "### Build image from source"
:end-before: "## Supported features"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "### Build image from source"
:end-before: "## Supported features"
:::

::::

:::::

## Supported features

:::::{tab-set}
:sync-group: device

::::{tab-item} NVIDIA CUDA
:sync: cuda

:::{include} cuda.inc.md
:start-after: "## Supported features"
:::

::::

::::{tab-item} AMD ROCm
:sync: rocm

:::{include} rocm.inc.md
:start-after: "## Supported features"
:::

::::

::::{tab-item} Intel XPU
:sync: xpu

:::{include} xpu.inc.md
:start-after: "## Supported features"
:::

::::

:::::
