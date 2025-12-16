<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单、快速且低成本的 LLM 部署服务，面向所有人
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>文档</b></a> | <a href="https://blog.vllm.ai/"><b>博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>论文</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

---
欢迎参加我们在旧金山举办的 [PyTorch Conference（10月22-23日）](https://events.linuxfoundation.org/pytorch-conference/) 和 [Ray Summit（11月3-5日）](https://www.anyscale.com/ray-summit/2025)，了解 vLLM 的最新动态并与团队见面！立即注册参加这一年一度最大的 vLLM 社区盛会！

---

*最新动态* 🔥

- [2025/11] 我们举办了 [vLLM 曼谷见面会](https://luma.com/v0f647nv)。我们与来自 Embedded LLM、AMD 和 Red Hat 的演讲者一起探讨了 vLLM、LMCache 推理以及低资源语言适配。会议幻灯片请见[此处](https://drive.google.com/drive/folders/1H0DS57F8HQ5q3kSOSoRmucPJWL3E0A_X?usp=sharing)。
- [2025/11] 我们在苏黎世举办了[首届 vLLM 欧洲见面会](https://luma.com/0gls27kb)，重点讨论量化、分布式推理以及大规模强化学习，演讲嘉宾来自 Mistral、IBM 和 Red Hat。会议幻灯片请见[此处](https://docs.google.com/presentation/d/1UC9PTLCHYXQpOmJDSFg6Sljra3iVXzc09DeEI7dnxMc/edit?usp=sharing)，录像请见[此处](https://www.youtube.com/watch?v=6m6ZE6yVEDI)。
- [2025/11] 我们举办了 [vLLM 北京见面会](https://mp.weixin.qq.com/s/xSrYXjNgr1HbCP4ExYNG1w)，聚焦于分布式推理以及 vLLM 对多样化加速器的支持！会议幻灯片请见[此处](https://drive.google.com/drive/folders/1nQJ8ZkLSjKxvu36sSHaceVXtttbLvvu-?usp=drive_link)。
- [2025/10] 我们举办了 [vLLM 上海见面会](https://mp.weixin.qq.com/s/__xb4OyOsImz-9eAVrdlcg)，专注于 vLLM 推理优化的实战分享！会议幻灯片请见[此处](https://drive.google.com/drive/folders/1KqwjsFJLfEsC8wlDugnrR61zsWHt94Q6)。
- [2025/09] 我们举办了 [vLLM 多伦多见面会](https://luma.com/e80e0ymm)，与来自 NVIDIA 和 Red Hat 的演讲者共同探讨大规模推理和投机解码（Speculative Decoding）！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1IYJYmJcu9fLpID5N5RbW_vO0XLo0CGOR14IXOjB61V8/edit?usp=sharing)。
- [2025/08] 我们举办了 [vLLM 深圳见面会](https://mp.weixin.qq.com/s/k8ZBO1u2_2odgiKWH_GVTQ)，聚焦 vLLM 周边的生态系统！会议幻灯片请见[此处](https://drive.google.com/drive/folders/1Ua2SVKVSu-wp5vou_6ElraDt2bnKhiEA)。
- [2025/08] 我们举办了 [vLLM 新加坡见面会](https://www.sginnovate.com/event/vllm-sg-meet)。我们与来自 Embedded LLM、AMD、WekaIO 和 A*STAR 的演讲者分享了 V1 版本的更新、分离式服务（Disaggregated Serving）以及多模态大模型（MLLM）的加速方案。会议幻灯片请见[此处](https://drive.google.com/drive/folders/1ncf3GyqLdqFaB6IeB834E5TZJPLAOiXZ?usp=sharing)。
- [2025/08] 我们举办了 [vLLM 上海见面会](https://mp.weixin.qq.com/s/pDmAXHcN7Iqc8sUKgJgGtg)，专注于 vLLM 的构建、开发与集成！会议幻灯片请见[此处](https://drive.google.com/drive/folders/1OvLx39wnCGy_WKq8SiVKf7YcxxYI3WCH)。
- [2025/05] vLLM 现已成为 PyTorch 基金会旗下的托管项目！公告详情请见[此处](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/)。
- [2025/01] 我们激动地宣布 vLLM V1 Alpha 版本发布：一次重大的架构升级，带来了 1.7 倍的速度提升！更整洁的代码、优化的执行循环、零开销的前缀缓存（Prefix Caching）、增强的多模态支持等等。详情请查看我们的[博客文章](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)。

<details>
<summary>往期新闻</summary>

- [2025/08] 我们与 Red Hat 和 Rebellions 联合举办了 [vLLM 韩国见面会](https://luma.com/cgcgprmh)！我们分享了 vLLM 的最新进展以及 vLLM 韩国社区的项目亮点。会议幻灯片请见[此处](https://drive.google.com/file/d/1bcrrAE1rxUgx0mjIeOWT6hNe2RefC5Hm/view)。
- [2025/08] 我们举办了 [vLLM 北京见面会](https://mp.weixin.qq.com/s/dgkWg1WFpWGO2jCdTqQHxA)，聚焦大规模 LLM 部署！会议幻灯片请见[此处](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF)，录像请见[此处](https://www.chaspark.com/#/live/1166916873711665152)。
- [2025/05] 我们举办了 [NYC vLLM 见面会](https://lu.ma/c1rqyf1f)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?usp=sharing)。
- [2025/04] 我们举办了 [亚洲开发者日](https://www.sginnovate.com/event/limited-availability-morning-evening-slots-remaining-inaugural-vllm-asia-developer-day)！vLLM 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/19cp6Qu8u48ihB91A064XfaXruNYiBOUKrBxAmDOllOo/edit?usp=sharing)。
- [2025/03] 我们举办了 [vLLM x Ollama 推理之夜](https://lu.ma/vllm-ollama)！vLLM 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/16T2PDD1YwRnZ4Tu8Q5r6n53c5Lr5c73UV9Vd2_eBo4U/edit?usp=sharing)。
- [2025/03] 我们举办了 [首届 vLLM 中国见面会](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg)！vLLM 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/1REHvfQMKGnvz6p3Fd23HhSO4c8j5WPGZV0bKYLwnHyQ/edit?usp=sharing)。
- [2025/03] 我们举办了 [美东 vLLM 见面会](https://lu.ma/7mu4k4xx)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1NHiv8EUFF1NLd3fEYODm56nDmL26lEeXCaDgyDlTsRs/edit#slide=id.g31441846c39_0_0)。
- [2025/02] 我们与 Meta 联合举办了 [第九期 vLLM 见面会](https://lu.ma/h7g3kuj9)！vLLM 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?usp=sharing)，AMD 的幻灯片请见[此处](https://drive.google.com/file/d/1Zk5qEJIkTmlQ2eQcXQZlljAx3m9s7nwn/view?usp=sharing)。Meta 的幻灯片暂未公布。
- [2025/01] 我们与 Google Cloud 联合举办了 [第八期 vLLM 见面会](https://lu.ma/zep56hui)！vLLM 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/1epVkt4Zu8Jz_S5OhEHPc798emsYh2BwYfRuDDVEF7u4/edit?usp=sharing)，Google Cloud 团队的幻灯片请见[此处](https://drive.google.com/file/d/1h24pHewANyRL11xy5dXUbvRC9F9Kkjix/view?usp=sharing)。
- [2024/12] vLLM 加入 [PyTorch 生态系统](https://pytorch.org/blog/vllm-joins-pytorch)！致力于为所有人提供简单、快速且低成本的 LLM 服务！
- [2024/11] 我们与 Snowflake 联合举办了 [第七期 vLLM 见面会](https://lu.ma/h0qvrajz)！vLLM 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing)，Snowflake 团队的幻灯片请见[此处](https://docs.google.com/presentation/d/1qF3RkDAbOULwz9WK5TOltt2fE9t6uIc_hVNLFAaQX6A/edit?usp=sharing)。
- [2024/10] 我们刚刚创建了一个专注于协调贡献和讨论功能的开发者 Slack ([slack.vllm.ai](https://slack.vllm.ai))。欢迎加入我们！
- [2024/10] Ray Summit 2024 为 vLLM 设立了特别分会场！vLLM 团队的开场演讲幻灯片请见[此处](https://docs.google.com/presentation/d/1B_KQxpHBTRa_mDF-tR6i8rWdOU5QoTZNcEg2MKZxEHM/edit?usp=sharing)。更多来自其他 vLLM 贡献者和用户的演讲请查看[此处](https://www.youtube.com/playlist?list=PLzTswPQNepXl6AQwifuwUImLPFRVpksjR)！
- [2024/09] 我们与 NVIDIA 联合举办了 [第六期 vLLM 见面会](https://lu.ma/87q3nvnh)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing)。
- [2024/07] 我们与 AWS 联合举办了 [第五期 vLLM 见面会](https://lu.ma/lp0gyjqr)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing)。
- [2024/07] 通过与 Meta 合作，vLLM 正式支持 Llama 3.1，包括 FP8 量化和流水线并行（Pipeline Parallelism）支持！请查看我们的[博客文章](https://blog.vllm.ai/2024/07/23/llama31.html)。
- [2024/06] 我们与 Cloudflare 和 BentoML 联合举办了 [第四期 vLLM 见面会](https://lu.ma/agivllm)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing)。
- [2024/04] 我们与 Roblox 联合举办了 [第三期 vLLM 见面会](https://robloxandvllmmeetup2024.splashthat.com/)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing)。
- [2024/01] 我们与 IBM 联合举办了 [第二期 vLLM 见面会](https://lu.ma/ygxbpzhl)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing)。
- [2023/10] 我们与 a16z 联合举办了 [首届 vLLM 见面会](https://lu.ma/first-vllm-meetup)！会议幻灯片请见[此处](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing)。
- [2023/08] 我们衷心感谢 [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) 为支持 vLLM 的开源开发和研究提供的慷慨资助。
- [2023/06] 我们正式发布了 vLLM！FastChat-vLLM 的集成自 4 月中旬起就已为 [LMSYS Vicuna 和 Chatbot Arena](https://chat.lmsys.org) 提供支持。详情请查阅我们的[博客文章](https://vllm.ai)。

</details>

---

## 关于 vLLM

vLLM 是一个快速且易于使用的 LLM（大型语言模型）推理和服务库。

vLLM 最初由加州大学伯克利分校的 [Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，现已演变成一个由学术界和工业界共同贡献的社区驱动项目。

vLLM 的**快速**体现在：

- 拥有 SOTA（State-of-the-art）的服务吞吐量
- 通过 [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) 高效管理 Attention 的键（Key）和值（Value）显存
- 对传入请求进行连续批处理（Continuous batching）
- 使用 CUDA/HIP Graph 快速执行模型
- 量化支持：[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[AutoRound](https://arxiv.org/abs/2309.05516)、INT4、INT8 和 FP8
- 优化的 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成
- 投机解码（Speculative decoding）
- 分块预填充（Chunked prefill）

vLLM 的**灵活易用**体现在：

- 与流行的 Hugging Face 模型无缝集成
- 支持多种解码算法的高吞吐量服务，包括*并行采样（parallel sampling）*、*集束搜索（beam search）*等
- 支持张量并行（Tensor）、流水线并行（Pipeline）、数据并行（Data）和专家并行（Expert）的分布式推理
- 支持流式输出（Streaming outputs）
- 提供兼容 OpenAI 的 API 服务器
- 支持 NVIDIA GPU、AMD CPU/GPU、Intel CPU/GPU、PowerPC CPU、Arm CPU 和 TPU。此外，还支持 Intel Gaudi、IBM Spyre 和 Huawei Ascend 等多种硬件插件
- 支持前缀缓存（Prefix caching）
- 支持多 LoRA（Multi-LoRA）功能

vLLM 无缝支持 Hugging Face 上大多数流行的开源模型，包括：

- Transformer 类 LLM（如 Llama）
- 混合专家（MoE）LLM（如 Mixtral、Deepseek-V2 和 V3）
- 嵌入模型（如 E5-Mistral）
- 多模态 LLM（如 LLaVA）

请在[此处](https://docs.vllm.ai/en/latest/models/supported_models.html)查看完整的支持模型列表。

## 快速上手

使用 `pip` 安装 vLLM 或[从源码构建](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)：

```bash
pip install vllm
```

访问我们的[文档](https://docs.vllm.ai/en/latest/)了解更多信息。

- [安装指南](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [快速入门](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

我们欢迎并重视任何形式的贡献与合作。
请查阅 [vLLM 贡献指南](https://docs.vllm.ai/en/latest/contributing/index.html) 了解如何参与。

## Sponsors

vLLM 是一个社区项目。我们的开发和测试计算资源由以下机构支持。感谢你们的支持！

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
现金捐赠：

- a16z
- Dropbox
- Sequoia Capital (红杉资本)
- Skywork AI
- ZhenFund (真格基金)

计算资源：

- Alibaba Cloud (阿里云)
- AMD
- Anyscale
- Arm
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Google Cloud
- Intel
- Lambda Lab
- Nebius
- Novita AI
- NVIDIA
- Replicate
- Roblox
- RunPod
- Trainy
- UC Berkeley
- UC San Diego
- Volcengine (火山引擎)

Slack 赞助商：Anyscale

我们还在 [OpenCollective](https://opencollective.com/vllm) 上设有官方的筹款渠道。我们计划使用这些资金来支持 vLLM 的开发、维护和推广。

## Citation

如果您在研究中使用了 vLLM，请引用我们的[论文](https://arxiv.org/abs/2309.06180)：

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
- 技术问题和功能请求，请使用 GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- 与其他用户讨论，请使用 [vLLM 论坛](https://discuss.vllm.ai)
- 协调贡献和开发工作，请使用 [Slack](https://slack.vllm.ai)
- 披露安全问题，请使用 GitHub 的 [安全公告 (Security Advisories)](https://github.com/vllm-project/vllm/security/advisories) 功能
- 商务合作与伙伴关系，请联系我们：[vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- 如果您希望使用 vLLM 的 Logo，请参考我们的 [Media Kit 仓库](https://github.com/vllm-project/media-kit)