# Apply SparseGPT to LLMs and deploy with nm-vllm

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/4c5jT1S)


This notebook walks through how to sparsify a pretrained LLM. To create a pruned model, you can leverage SparseGPT. Quantizing reduces the model's precision from FP16 to INT4 which effectively reduces the file size by ~70%. The main benefits are lower latency and memory usage.

This notebook requires an NVIDIA GPU with compute capability >= 8.0 (>=Ampere) because of Marlin kernel restrictions. This will not run on T4 or V100.
