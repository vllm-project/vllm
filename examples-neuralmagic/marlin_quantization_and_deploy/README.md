# Performantly Quantize LLMs to 4-bits with Marlin and nm-vllm

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/3uY6NTx)

This notebook walks through how to compress a pretrained LLM and deploy it with `nm-vllm`. To create a new 4-bit quantized model, we can leverage AutoGPTQ. Quantizing reduces the model's precision from FP16 to INT4 which effectively reduces the file size by ~70%. The main benefits are lower latency and memory usage.

Developed in collaboration with IST-Austria, [GPTQ](https://arxiv.org/abs/2210.17323) is the leading quantization algorithm for LLMs, which enables compressing the model weights from 16 bits to 4 bits with limited impact on accuracy. [nm-vllm](https://github.com/neuralmagic/nm-vllm) includes support for the recently-developed Marlin kernels for accelerating GPTQ models. Prior to Marlin, the existing kernels for INT4 inference failed to scale in scenarios with multiple concurrent users.

This notebook requires an NVIDIA GPU with compute capability >= 8.0 (>=Ampere) because of Marlin kernel restrictions. This will not run on T4 or V100 currently. This was tested on an A100 on Colab.
