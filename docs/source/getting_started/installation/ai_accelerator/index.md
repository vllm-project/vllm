# AI Accelerator

vLLM is a Python libary that supports the following AI accelerators. Select your AI accelerator type to see vendor specific instructions:

::::{tab-set}
:sync-group: device

:::{tab-item} TPU
:sync: tpu

```{include} tpu.md
:start-after: "# Installation"
:end-before: "## Requirements"
```

:::

:::{tab-item} Intel Gaudi
:sync: hpu-gaudi

```{include} hpu-gaudi.md
:start-after: "# Installation"
:end-before: "## Requirements"
```

:::

:::{tab-item} Neuron
:sync: neuron

```{include} neuron.md
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

:::{tab-item} TPU
:sync: tpu

```{include} tpu.md
:start-after: "## Requirements"
:end-before: "## Install released versions"
```

:::

:::{tab-item} Intel Gaudi
:sync: hpu-gaudi

```{include} hpu-gaudi.md
:start-after: "## Requirements"
:end-before: "## Option 1: Build from source with docker (recommended)"
```

:::

:::{tab-item} Neuron
:sync: neuron

```{include} neuron.md
:start-after: "## Requirements"
:end-before: "## Quick start using Dockerfile"
```

:::

::::