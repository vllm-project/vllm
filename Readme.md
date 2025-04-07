# CPEN 511 Project:

Presented by:
- Tom Wang
- Jiajun Huang

from the University of British Columbia

## Introduction

This work is based on the implementatino of vllm open source project. The [original readme](Original_README.md) can be found in the same directory. The offical website of vllm is [here](https://vllm.ai/).

## Getting Started

To install vllm, you can use the following command:

```bash
git clone git@github.com:CPEN511-Project-2024W2-UBC/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
for NVIDIA CUDA users. For other options, check the [official installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).However, no test has been done on non-NVIDIA CUDA environment for our project.

## Running the code

To run a simple test to demenstrate our work, you can use the following command:

```bash
cd playground
TODO!!!!!
```

## Naive Implementation

Let $n$ be number of sequences schduled in KV cache. The naive implementation of the KV cache is to assume that the probability of a sequence that needs to be appended to be $\frac{1}{16}$, then we do a search of how many blocks each sequence will need to append. That is, let $o\in\mathbb{R}$, we test $o$ till we find an optimal solution. 

For pratical reason, we test $o \in [0,4]$ with step size $0.1$. The result is shown in the following figure: