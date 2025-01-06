(installation-arm)=

# Installation for ARM CPUs

vLLM has been adapted to work on ARM64 CPUs with NEON support, leveraging the CPU backend initially developed for the x86 platform. This guide provides installation instructions specific to ARM. For additional details on supported features, refer to the [x86 CPU documentation](#installation-x86) covering:

- CPU backend inference capabilities
- Relevant runtime environment variables
- Performance optimization tips

ARM CPU backend currently supports Float32, FP16 and BFloat16 datatypes.
Contents:

1. [Requirements](#arm-backend-requirements)
2. [Quick Start with Dockerfile](#arm-backend-quick-start-dockerfile)
3. [Building from Source](#build-arm-backend-from-source)

(arm-backend-requirements)=

## Requirements

- **Operating System**: Linux or macOS
- **Compiler**: `gcc/g++ >= 12.3.0` (optional, but recommended)
- **Instruction Set Architecture (ISA)**: NEON support is required

(arm-backend-quick-start-dockerfile)=

## Quick Start with Dockerfile

You can quickly set up vLLM on ARM using Docker:

```console
$ docker build -f Dockerfile.arm -t vllm-cpu-env --shm-size=4g .
$ docker run -it \
             --rm \
             --network=host \
             --cpuset-cpus=<cpu-id-list, optional> \
             --cpuset-mems=<memory-node, optional> \
             vllm-cpu-env
```

(build-arm-backend-from-source)=

## Building from Source

To build vLLM from source on Ubuntu 22.04 or other Linux distributions, follow a similar process as with x86. Testing has been conducted on AWS Graviton3 instances for compatibility.
