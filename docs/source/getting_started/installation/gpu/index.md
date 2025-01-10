# GPU

vLLM is a Python libary that supports the following GPU variants. Select your GPU type to see vendor specific instructions:

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda

```{include} cuda.md
:start-after: "# Installation"
:end-before: "## Requirements"
```

:::

:::{tab-item} ROCm
:sync: rocm

```{include} rocm.md
:start-after: "# Installation"
:end-before: "## Requirements"
```

:::

:::{tab-item} XPU
:sync: xpu

```{include} xpu.md
:start-after: "# Installation"
:end-before: "## Requirements"
```

:::

::::

## Requirements

- OS: Linux
- Python: 3.9 -- 3.12

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda

```{include} cuda.md
:start-after: "## Requirements"
:end-before: "## Python"
```

:::

:::{tab-item} ROCm
:sync: rocm

```{include} rocm.md
:start-after: "## Requirements"
:end-before: "## Python"
```

:::

:::{tab-item} XPU
:sync: xpu

```{include} xpu.md
:start-after: "## Requirements"
:end-before: "## Python"
```

:::

::::

## Python

### Create a new Python environment

```{include} ../python_env_setup.md
```

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda

```{include} cuda.md
:start-after: "## Create a new Python environment"
:end-before: "### Pre-built wheels"
```

:::

::::

### Pre-built wheels

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda
```{include} cuda.md
:start-after: "### Pre-built wheels"
:end-before: "### Build wheel from source"
```
:::

:::{tab-item} ROCm
:sync: rocm
```{include} rocm.md
:start-after: "### Pre-built wheels"
:end-before: "### Build wheel from source"
```
:::

:::{tab-item} XPU
:sync: xpu
```{include} xpu.md
:start-after: "### Pre-built wheels"
:end-before: "### Build wheel from source"
```
:::

::::

(build-from-source)=

### Build wheel from source

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda
```{include} cuda.md
:start-after: "### Build wheel from source"
:end-before: "## Docker"
```
:::

:::{tab-item} ROCm
:sync: rocm
```{include} rocm.md
:start-after: "### Build wheel from source"
:end-before: "## Docker"
```
:::

:::{tab-item} XPU
:sync: xpu
```{include} xpu.md
:start-after: "### Build wheel from source"
:end-before: "## Docker"
```
:::

::::

## Docker

### Pre-built images

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda
```{include} cuda.md
:start-after: "### Pre-built images"
:end-before: "### Build image from source"
```
:::

:::{tab-item} ROCm
:sync: rocm
```{include} rocm.md
:start-after: "### Pre-built images"
:end-before: "### Build image from source"
```
:::

:::{tab-item} XPU
:sync: xpu
```{include} xpu.md
:start-after: "### Pre-built images"
:end-before: "### Build image from source"
```
:::

::::

### Build image from source

::::{tab-set}
:sync-group: device

:::{tab-item} CUDA
:sync: cuda
```{include} cuda.md
:start-after: "### Build image from source"
:end-before: "## Extra information"
```
:::

:::{tab-item} ROCm
:sync: rocm
```{include} rocm.md
:start-after: "### Build image from source"
:end-before: "## Extra information"
```
:::

:::{tab-item} XPU
:sync: xpu
```{include} xpu.md
:start-after: "### Build image from source"
:end-before: "## Extra information"
```
:::

::::

## Supported features

See <project:#feature-x-hardware> compatibility matrix for CUDA and ROCm feature support information.

::::{tab-set}
:sync-group: device

:::{tab-item} XPU
:sync: xpu
```{include} xpu.md
:start-after: "## Supported features"
```
:::

::::