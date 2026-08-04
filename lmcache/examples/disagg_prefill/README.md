# Disaggregated Prefill Examples for LMCache with vLLM v1

This directory contains examples demonstrating how to run LMCache with disaggregated prefill using NIXL. Disaggregated prefill allows you to separate the prefill (prompt processing) and decode (token generation) phases of LLM inference across different GPU instances, enabling better resource utilization and scalability.

## Overview

Disaggregated prefill architecture separates the compute-intensive prefill phase from the memory-intensive decode phase:

- **Prefill servers**: Handle prompt processing and KV cache generation
- **Decode server**: Handles token generation using cached KV states
- **Proxy server**: Coordinates requests between prefill and decode servers

This architecture provides several benefits:
- Better GPU utilization by matching workload characteristics to hardware
- Improved scalability by independently scaling prefill and decode capacity
- Reduced latency through parallel processing
- Cost optimization by using different instance types for different phases

## Available Examples

### 1p1d - Single Prefill, Single Decode
Directory: [`1p1d/`](./1p1d/)

**Requirements**: At least 2 GPUs

This is the simplest configuration to get started with disaggregated prefill.

### xpyd - Multiple Prefill, Multiple Decode
Directory: [`xpyd/`](./xpyd/)

**Requirements**: At least 3 GPUs

This configuration demonstrates multiple prefills and multiple decodes.

## Prerequisites

Before running any example, ensure you have:

- [LMCache](https://github.com/LMCache/LMCache) installed: `pip install lmcache`
- [NIXL](https://github.com/ai-dynamo/nixl) installed
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct
- Sufficient GPU resources (see individual example requirements)

## Quick Start & Benchmarking

1. Choose the appropriate example based on your GPU resources:
   - For 2 GPUs: Use [`1p1d/`](./1p1d/)
   - For 3+ GPUs: Use [`xpyd/`](./xpyd/)

2. Navigate to the chosen directory:
   ```bash
   cd 1p1d/  # or cd xpyd/
   ```

3. Follow the specific README instructions in that directory


## Architecture Components

Each example includes:

- **Main script**: `disagg_example_*.sh` - Main entry point to run the example
- **Launcher script**: `disagg_vllm_launcher.sh` - Launches vLLM servers and proxy
- **Proxy server**: `disagg_proxy_server.py` - FastAPI server coordinating requests
- **Configuration files**: YAML configs for prefill and decode servers
- **Log files**: Generated during execution for debugging

## Troubleshooting

- **GPU Memory Issues**: Ensure you have sufficient VRAM for the model on each GPU
- **Port Conflicts**: Check that ports 8100, 8101, 8200, and 9000 are available
- **HF Token**: Verify your Hugging Face token has access to Llama 3.1 models
- **Dependencies**: Ensure both LMCache and NIXL are properly installed

For detailed troubleshooting, check the log files generated in each example directory.

## Further Reading

- [LMCache Documentation](https://github.com/LMCache/LMCache)
- [NIXL Documentation](https://github.com/ai-dynamo/nixl)
- [vLLM Documentation](https://docs.vllm.ai/) 