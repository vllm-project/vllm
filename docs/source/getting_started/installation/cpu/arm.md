# Installation

vLLM has been adapted to work on ARM64 CPUs with NEON support, leveraging the CPU backend initially developed for the x86 platform.

ARM CPU backend currently supports Float32, FP16 and BFloat16 datatypes.

## Requirements

- OS: Linux
- Compiler: `gcc/g++ >= 12.3.0` (optional, recommended)
- Instruction Set Architecture (ISA): NEON support is required


## Python

### Pre-built wheels

### Build wheel from source

:::{include} build.md
:::

Testing has been conducted on AWS Graviton3 instances for compatibility.

## Docker

### Pre-built images

### Build image from source

## Supported features