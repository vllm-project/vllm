# DGX Spark 问题记录

## 问题 #1: PyTorch SM 12.1 (GB10) 兼容性警告

**现象：**
```
UserWarning: GPU 0 (NVIDIA GB10) has compute capability 12.1 which is not supported by this version of PyTorch.
PyTorch only supports compute capabilities 8.0-9.0.
```

**复现步骤：**
1. 在 DGX Spark (GB10) 上运行任何 PyTorch CUDA 操作
2. `python -c "import torch; print(torch.cuda.is_available())"`

**环境：**
- DGX Spark 版本：NVIDIA DGX Spark
- GPU: NVIDIA GB10 (SM 12.1)
- CUDA 版本：13.0
- PyTorch 版本：2.10.0+cu126
- Driver: 580.95.05

**可能原因：**
- PyTorch 2.10 官方只支持 SM 8.0-9.0 (A100, H100, H200)
- GB10 (Blackwell 架构, SM 12.1) 是更新的硬件
- 需要等待 PyTorch 2.11+ 或从源码构建支持 SM 12.1 的 PyTorch

**解决方案：**
- 当前 workaround: 忽略 warning，CUDA 仍然可用
- 长期方案：等待 PyTorch 2.11+ 官方支持 SM 12.1
- 或从源码构建 PyTorch with `-D TORCH_CUDA_ARCH_LIST="12.1"`

**影响范围：**
- 仅 DGX Spark / GB10 用户
- 性能影响：某些 kernel 可能无法使用最优路径
- 非阻塞性：CUDA 仍然可用，vLLM 可以运行

---

## 问题 #2: MoE FP8 Marlin 需要手动启用

**现象：**
- 在 DGX Spark 上运行 NVFP4 MoE 模型时，默认使用 FlashInfer CUTLASS backend
- 需要手动设置环境变量才能使用 Marlin：
  ```bash
  export VLLM_TEST_FORCE_FP8_MARLIN=1
  export VLLM_NVFP4_GEMM_BACKEND=marlin
  ```

**复现步骤：**
1. 在 DGX Spark 上运行 NVFP4 MoE 模型
2. 观察日志中的 backend 选择

**环境：**
- DGX Spark: NVIDIA GB10 (SM 12.1)
- vLLM: dev 版本
- 模型：Nemotron-3-Super-120B-A12B-NVFP4 等

**可能原因：**
- vLLM 没有自动检测 DGX Spark / SM 12.1
- 默认 backend 选择逻辑没有考虑 GB10 的特殊性
- SM 12.1 只有 101KB SMEM（vs SM100 的 228KB），CUTLASS FP4 路径需要额外 patch

**解决方案：**
- **短期：** 手动设置上述环境变量
- **长期（PR 方向）：** 在 `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` 的 `select_nvfp4_moe_backend()` 中添加自动检测：
  ```python
  major, minor = torch.cuda.get_device_capability()
  if major == 12 and minor == 1:  # GB10 / DGX Spark
      return "marlin"
  ```

**影响范围：**
- 仅 DGX Spark / GB10 用户
- 性能影响：Marlin vs CUTLASS 性能差异极小（~7% single request, 几乎相同 concurrent）
- 用户体验：Marlin 开箱即用，省去 5+ 分钟 JIT 编译

**参考：**
- Issue #37141: [Feature] Upstream DGX Spark improvements
- rmagur1203 的基准测试报告（144 组合）
- ProExpertProg (maintainer) 确认："just need to enable marlin by default when we detect DGX spark?"

---

## 问题 #3: aarch64 架构安装问题

**现象：**
- 在 DGX Spark (arm64/aarch64) 上安装 vLLM 可能遇到依赖问题
- 某些 wheel 包只有 x86_64 版本

**环境：**
- 架构：aarch64 (arm64)
- 系统：NVIDIA DGX Spark OS

**可能原因：**
- PyPI 上某些包没有 aarch64 wheel
- 需要从源码构建

**解决方案：**
- 使用社区维护的 Docker 镜像（如 eugr 的 spark-vllm-docker）
- 或参考 PR #38057 (bbrowning) 的 aarch64 dev setup 改进

**影响范围：**
- 仅 aarch64 / DGX Spark 用户
- 阻塞性：影响开发环境搭建

---

## 问题 #4: FlashInfer + MTP Speculative Decoding 在 SM121 崩溃

**现象：**
- 使用 FlashInfer backend + MTP=2 speculative decoding 在 GB10 上出现 illegal memory access
- Issue #37754

**复现步骤：**
1. 在 DGX Spark 上运行 MTP speculative decoding
2. 设置 MTP=2 或更高

**环境：**
- GPU: NVIDIA GB10 (SM 12.1)
- FlashInfer backend

**可能原因：**
- FlashInfer kernel 在 SM 12.1 上的兼容性问题
- 需要 FlashInfer 层面的修复

**解决方案：**
- 短期 workaround: 降低 MTP 值或切换到其他 backend
- 长期：需要 FlashInfer 项目修复

**影响范围：**
- 仅 SM 12.1 (GB10) 用户
- 阻塞性：影响 speculative decoding 功能
- 状态：已有人在跟进 (@bzubs, @rmagur1203)

---

## 总结

**DGX Spark 独特优势：**
1. 我们有真实的 GB10 硬件可以测试
2. 社区里 DGX Spark 用户相对少，贡献更容易被注意到
3. 维护者 (ProExpertProg) 明确表示需要 DGX Spark 相关的改进

**最佳 PR 方向：**
- #37141: 自动检测 GB10/SM121 并默认启用 Marlin backend
- 改动小，逻辑清晰
- 有 maintainer 认可
- 可以直接在本地测试

**环境记录：**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- CUDA: 13.0
- PyTorch: 2.10.0+cu126 (with SM 12.1 warning)
- vLLM: dev version from source
