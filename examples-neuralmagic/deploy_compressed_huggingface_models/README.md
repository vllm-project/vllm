# Deploy Compressed LLMs from Hugging Face with nm-vllm

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/nm-vllm/blob/main/examples-neuralmagic/deploy_compressed_huggingface_models/Deploy_Compressed_LLMs_from_Hugging_Face_with_nm_vllm.ipynb)


This notebook walks through how to deploy compressed models with nm-vllm's latest memory and performance optimizations.

Neural Magic maintains a variety of compressed models on our Hugging Face organization profiles, [neuralmagic](https://huggingface.co/neuralmagic) and [nm-testing](https://huggingface.co/nm-testing). A collection of ready-to-use compressed models is available [here](https://huggingface.co/collections/neuralmagic/compressed-llms-for-nm-vllm-65e73e3d51d3200e34b77431).

For unstructured sparsity, NVIDIA GPUs with compute capability >= 7.0 (V100, T4, A100) is required. For semi-structured sparsity or Marlin quantized kernels, a NVIDIA GPU with compute capability >= 8.0 (>=Ampere, A100) is required. This was tested on an A100 on Colab.