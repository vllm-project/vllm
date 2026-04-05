<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单、快速、低成本的大语言模型推理服务
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>文档</b></a> | <a href="https://blog.vllm.ai/"><b>博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>论文</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

<p align="center">
<a href="https://github.com/vllm-project/vllm#readme">English</a> | <b>简体中文</b>
</p>

---

## 关于 vLLM

vLLM 是一个快速且易用的大语言模型（LLM）推理与服务库。

vLLM 最初由加州大学伯克利分校 [Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，现已发展为一个由社区驱动的项目，汇集了学术界和工业界的广泛贡献。

### 高性能

- 业界领先的服务吞吐量
- 基于 [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) 的高效注意力键值缓存管理
- 传入请求的连续批处理（Continuous Batching）
- 基于 CUDA/HIP Graph 的快速模型执行
- 多种量化方案：[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[AutoRound](https://arxiv.org/abs/2309.05516)、INT4、INT8 和 FP8
- 高性能 CUDA 算子，集成 FlashAttention 和 FlashInfer
- 推测解码（Speculative Decoding）
- 分块预填充（Chunked Prefill）

### 灵活易用

- 与主流 Hugging Face 模型无缝集成
- 支持多种解码算法的高吞吐服务，包括*并行采样*、*束搜索*等
- 支持张量并行、流水线并行、数据并行和专家并行的分布式推理
- 流式输出
- 兼容 OpenAI 的 API 服务器
- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、Arm CPU 以及 TPU，还支持 Intel Gaudi、IBM Spyre、华为昇腾等多种硬件插件
- 前缀缓存（Prefix Caching）
- Multi-LoRA 支持

### 支持的模型

vLLM 支持 HuggingFace 上大多数主流开源模型，包括：

- Transformer 类 LLM（如 Llama）
- 混合专家（MoE）LLM（如 Mixtral、Deepseek-V2 和 V3）
- 嵌入模型（如 E5-Mistral）
- 多模态 LLM（如 LLaVA）

完整支持模型列表请参阅[这里](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 快速开始

使用 `pip` 安装 vLLM，或[从源码构建](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)：

```bash
pip install vllm
```

访问[官方文档](https://docs.vllm.ai/en/latest/)了解更多。

- [安装指南](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [快速入门](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)

## 参与贡献

我们欢迎并重视任何形式的贡献与合作。请查阅[贡献指南](https://docs.vllm.ai/en/latest/contributing/index.html)了解如何参与。

## 引用

如果您在研究中使用了 vLLM，请引用我们的[论文](https://arxiv.org/abs/2309.06180)：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系我们

<!-- --8<-- [start:contact-us] -->
- 技术问题和功能请求，请使用 GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- 与其他用户交流，请访问 [vLLM 论坛](https://discuss.vllm.ai)
- 协调贡献和开发，请加入 [Slack](https://slack.vllm.ai)
- 安全问题披露，请使用 GitHub 的[安全公告](https://github.com/vllm-project/vllm/security/advisories)功能
- 合作与伙伴关系，请联系 [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## 媒体素材

- 如需使用 vLLM 的 Logo，请参阅[媒体素材仓库](https://github.com/vllm-project/media-kit)
