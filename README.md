<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## vLLM for Windows

vLLM for Windows build & kernels. This repository will be updated when new versions of vLLM are released.

**Don't open a new Issue to request a specific commit build. Wait for a new stable release.**

**Don't open Issues for general vLLM questions or non Windows related problems. Only Windows specific issues.** Any Issue opened that is not Windows specific will be closed automatically.

**Don't request a wheel for your specific environment.** If your environment does not match the released wheel, build your own wheel from source by following the [instructions below](https://github.com/SystemPanic/vllm-windows?tab=readme-ov-file#building-from-source).

### Windows instructions:

#### Installing an existing release wheel:

1. Ensure that you have the correct Python, Torch and CUDA version of the wheel. The Python, Torch and CUDA version of the wheel is specified in the release version.
2. Download the wheel from the release version of your preference (latest wheel [here](https://github.com/SystemPanic/vllm-windows/releases/latest)).
3. Install it with ```pip install DOWNLOADED_WHEEL_PATH```

#### Building from source:

A Visual Studio 2019 or newer is required to launch the compiler x64 environment. The installation path is referred in the instructions as VISUAL_STUDIO_INSTALL_PATH.

CUDA path will be found automatically if you have the bin folder in your PATH, or have the CUDA installation path settled on well-known environment vars like CUDA_ROOT, CUDA_HOME or CUDA_PATH.

If none of these are present, make sure to set the environment variable before starting the build:
set CUDA_ROOT=CUDA_INSTALLATION_PATH

1. Open a Command Line (cmd.exe)
2. **Clone the vLLM for Windows repository from vllm-for-windows branch (NOT MAIN): ```cd C:\ & git clone --single-branch --branch vllm-for-windows https://github.com/SystemPanic/vllm-windows.git```**
3. Execute (in cmd) ```VISUAL_STUDIO_INSTALL_PATH\VC\Auxiliary\Build\vcvarsall.bat x64```
4. Change the working directory to the cloned repository path, for example: ```cd C:\vllm-windows```
5. Set the following environment variables:

```
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

#Optional variables:

#To include cuDSS (only if you have cuDSS installed)
set USE_CUDSS=1
set CUDSS_LIBRARY_PATH=PATH_TO_CUDSS_INSTALL_DIR\lib\12
set CUDSS_INCLUDE_PATH=PATH_TO_CUDSS_INSTALL_DIR\include

#To include cuSPARSELt (only if you have cuSPARSELt installed)
set USE_CUSPARSELT=1
set CUSPARSELT_INCLUDE_PATH=PATH_TO_CUSPARSELT_INSTALL_DIR\include
set CUSPARSELT_LIBRARY_PATH=PATH_TO_CUSPARSELT_INSTALL_DIR\lib

#To include cuDNN:
set USE_CUDNN=1
set CUDNN_LIBRARY_PATH=PATH_TO_CUDNN_INSTALL_DIR\lib\CUDNN_CUDA_VERSION\x64
set CUDNN_INCLUDE_PATH=PATH_TO_CUDNN_INSTALL_DIR\include\CUDNN_CUDA_VERSION

#Flash Attention v3 build has been disabled inside WSL2 and Windows due to compiler being killed on WSL2, and extremely long compiling times on Windows. Hopper is not available on Windows, so FA3 has no sense anyway. 
#Build can be forcefully enabled using the following environment var:
set VLLM_FORCE_FA3_WINDOWS_BUILD=1

```
6. Build & install:
```
#With torch 2.11 cuda 12.6 (change cu126 with your installed CUDA version)
pip install --pre torch==2.11.0.dev20260216+cu126 torchvision==0.26.0.dev20260216+cu126 torchaudio==2.11.0.dev20260216+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

#With your already installed torch cuda version (make sure you have torch cuda installed if you use a virtual environment)
python use_existing_torch.py

pip install -r requirements/build.txt
pip install -r requirements/windows.txt
pip install . --no-build-isolation

```

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
