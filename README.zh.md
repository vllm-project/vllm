<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
让每个人都能享受简易、极速且低成本的大语言模型推理服务
</h3>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
| <a href="https://docs.vllm.ai"><b>官方文档</b></a> | <a href="https://blog.vllm.ai/"><b>技术博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>学术论文</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

🔥 我们推出了全新的 vLLM 官方网站，帮助您快速上手 vLLM。欢迎访问 [vllm.ai](https://vllm.ai) 了解详情。
如需参加社区活动，请访问 [vllm.ai/events](https://vllm.ai/events) 加入我们。

---

## 关于 vLLM

vLLM 是一个用于大语言模型（LLM）推理与高吞吐服务的快速、易用开源框架。

vLLM 最早由加州大学伯克利分校（UC Berkeley）的 [Sky Computing Lab](https://sky.cs.berkeley.edu) 团队开发，如今已成长为全球最活跃的开源 AI 项目之一，由来自全球数十所顶尖学术机构、科技企业及 2000 多位贡献者组成的多元化社区共同构建与维护。

### 🚀 极致性能表现：

- **业界顶尖的吞吐性能**（State-of-the-art serving throughput）。
- **PagedAttention 核心机制**：高效管理注意力机制中的 Key-Value 内存（KV Cache），彻底解决显存碎片化瓶颈（[论文解读](https://blog.vllm.ai/2023/06/20/vllm.html)）。
- **持续请求批处理（Continuous Batching）**、分块预填充（Chunked Prefill）与前缀缓存（Prefix Caching）。
- **分段与完整 CUDA/HIP Graphs**：实现快速且灵活的模型计算图执行。
- **全方位量化加速**：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO 及[更多量化方案](https://docs.vllm.ai/en/latest/features/quantization/index.html)。
- **高度优化的注意力算子内核**：包括 FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA 和 Triton。
- **优化的 GEMM / MoE 内核**：使用 CUTLASS、TRTLLM-GEN、CuTeDSL 适配不同计算精度。
- **投机解码（Speculative Decoding）**：支持 n-gram、suffix、EAGLE、DFlash 等推测加速算法。
- **自动化算子生成与图优化**：基于 `torch.compile` 实现图级别自动变换。
- **PD 分离架构（Disaggregated Prefill, Decode, and Encode）**：实现 Prefill 与 Decode 阶段的独立硬件解耦部署。

### 🛠️ 灵活易用的工程体验：

- **无缝对接 Hugging Face 社区模型**。
- **支持多种高效解码策略**：包含并行采样（Parallel Sampling）、束搜索（Beam Search）等。
- **多维度分布式推理并行支持**：张量并行（TP）、流水线并行（PP）、数据并行（DP）、专家并行（EP）与上下文并行（CP）。
- **流式输出（Streaming Outputs）**。
- **结构化输出生成**：集成 xgrammar 与 guidance 引擎，保障 JSON 等强类型格式输出。
- **原生工具调用（Tool Calling）与推理思考过程解析器（Reasoning Parsers）**。
- **OpenAI 兼容 API 服务端**，同时支持 Anthropic Messages API 与高性能 gRPC 通信。
- **高效 Multi-LoRA 动态适配**：同时支持 Dense 稠密层与 MoE 混合专家层。
- **跨平台多硬件生态支持**：支持 NVIDIA GPU、AMD GPU、Intel GPU 以及 x86/ARM/PowerPC CPU；并通过硬件插件广泛支持 Google TPU、Intel Gaudi、IBM Spyre、华为昇腾（Huawei Ascend）、Rebellions NPU、Apple Silicon、沐曦（MetaX）GPU 等异构算力。

### 🌟 广泛支持 200+ 种 Hugging Face 模型架构：

- **Decoder-only 经典模型**（如 Llama、Qwen、Gemma 等）
- **MoE 混合专家模型**（如 Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS 等）
- **混合注意力与状态空间模型**（如 Mamba、Qwen3.5 等）
- **多模态视觉模型 (VLM)**（如 LLaVA、Qwen-VL、Pixtral 等）
- **向量嵌入与检索模型**（如 E5-Mistral、GTE、ColBERT 等）
- **奖励与分类评估模型**（如 Qwen-Math 等）

👉 查看[完整支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)。

---

## 快速上手 (Getting Started)

推荐使用 [`uv`](https://docs.astral.sh/uv/)（推荐）或 `pip` 安装 vLLM：

```bash
uv pip install vllm
```

如需开发调试，也可从源码编译：[源码编译指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)。

欢迎访问[官方完整文档](https://docs.vllm.ai/en/latest/)了解更多详情：

- [安装部署指南 (Installation)](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [快速入门教程 (Quickstart)](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [完整支持模型列表 (Supported Models)](https://docs.vllm.ai/en/latest/models/supported_models.html)

---

## 参与贡献 (Contributing)

我们非常欢迎并高度重视来自开源社区的任何贡献与合作。
请查阅 [vLLM 贡献指南](https://docs.vllm.ai/en/latest/contributing/index.html) 了解如何参与代码提交与功能开发。

## 论文引用 (Citation)

如果您在学术研究中使用了 vLLM，请引用我们的 [SOSP '23 学术论文](https://arxiv.org/abs/2309.06180)：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系我们 (Contact Us)

<!-- --8<-- [start:contact-us] -->
- 技术疑问与功能需求：请在 GitHub [Issues](https://github.com/vllm-project/vllm/issues) 中提出讨论
- 社区用户交流：欢迎加入 [vLLM 官方论坛](https://discuss.vllm.ai)
- 协同开发与代码贡献：欢迎加入 [vLLM 开发者 Slack](https://slack.vllm.ai)
- 安全漏洞披露：请通过 GitHub 的 [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) 渠道提交
- 商业合作与生态伙伴：请发送邮件至 [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## 媒体资源包 (Media Kit)

- 如需使用 vLLM 官方 Logo 与品牌标识，请参考我们的 [Media Kit 仓库](https://github.com/vllm-project/media-kit)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年8月31日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
