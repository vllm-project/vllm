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

## 2026-03-26 10:29 更新：NVIDIA 官方 DGX Spark 修复进展

**PR #38126** — "[NVIDIA] Fix DGX Spark logic" by johnnynunez (NVIDIA maintainer)
- **状态：** Open，7 条评论，活跃讨论中
- **Labels：** `ready`, `ci/build`, `nvidia`
- **最后更新：** 2026-03-25T21:12:49Z (~13h 前)
- **关键讨论：**
  - eugr 实测：编译成功，包含 nvfp4 kernels，auto tuner errors 正常
  - RobTand 解释：auto tuner errors 正常（block sizes 少）
  - johnnynunez 已 push 更新 ("done @mgoin")
- **我的策略：** 等 merge 后检查代码变更，确认是否覆盖 SM 12.1 Marlin 自动检测
  - 如已覆盖 → 转向其他 DGX Spark 方向（部署文档、性能优化）
  - 如未覆盖 → 继续推进 #37141

**教训：**
- NVIDIA 维护者也在关注 DGX Spark，这是好事
- 社区协作：我可以测试他们的修复，提供反馈
- 避免重复工作：先看看 #38126 解决了什么

---

## 2026-03-26 10:29 Cron 检查摘要

**PR #37959 (我的 Helm chart 修复):**
- 状态：Open ~78h
- Labels: `bug`, `documentation` (仍无 `ready`)
- 人类 review: 0
- 下一步：考虑在 Slack #pr-reviews 礼貌求助

**PR #38126 (NVIDIA DGX Spark 修复):**
- 状态：Open，~13h 无更新
- Labels: `ready`, `ci/build`, `nvidia`
- 等待 merge 后检查代码

**izhuhaoran 动态:**
- 5 open PRs，无新 merge
- 专注 Blackwell/Model Runner V2 方向

---

## 总结

**DGX Spark 独特优势：**
1. 我们有真实的 GB10 硬件可以测试
2. 社区里 DGX Spark 用户相对少，贡献更容易被注意到
3. 维护者 (ProExpertProg, johnnynunez) 明确表示需要 DGX Spark 相关的改进

**最佳 PR 方向：**
- #37141: 自动检测 GB10/SM121 并默认启用 Marlin backend（如果 #38126 没覆盖）
- 改动小，逻辑清晰
- 有 maintainer 认可
- 可以直接在本地测试

**环境记录：**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- CUDA: 13.0
- PyTorch: 2.10.0+cu126 (with SM 12.1 warning)
- vLLM: dev version from source

---

## 2026-03-26 12:38 Cron 检查摘要

**PR #37959 (我的 Helm chart 修复):**
- 状态：Open ~56h
- Labels: `bug`, `documentation` (仍无 `ready`)
- 人类 review: 0
- 下一步：在 Slack #pr-reviews 礼貌求助

**PR #38126 (NVIDIA DGX Spark 修复):**
- 状态：Open，~15h 无更新
- Labels: `ready`, `ci/build`, `nvidia`
- 等待 merge 后检查代码是否覆盖 SM 12.1 Marlin 自动检测

**izhuhaoran 动态:**
- 5 open PRs，无新 merge
- 专注 Model Runner V2、HMA、Blackwell 方向
- 节奏稳定，非批量提交

**本地环境:**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- 分支：fix/helm-chart-selector-labels
- 未提交更改：DAILY_LEARNINGS.md, dgx_spark_issues.md

**DGX Spark 独特优势确认:**
- 真实 GB10 硬件可测试
- 社区 DGX Spark 用户少，贡献易被注意
- 维护者 (johnnynunez, ProExpertProg) 明确需要 DGX Spark 改进

---

## 2026-03-26 13:45 Cron 检查摘要

**PR #37959 (我的 Helm chart 修复):**
- 状态：Open ~57h
- Labels: `bug`, `documentation` (仍无 `ready`)
- 人类 review: 0
- 下一步：在 Slack #pr-reviews 礼貌求助

**PR #38126 (NVIDIA DGX Spark 修复):**
- 状态：Open，~16h 无更新
- 作者：johnnynunez (NVIDIA maintainer)
- Labels: `ready`, `ci/build`, `nvidia`
- 等待 merge 后检查代码是否覆盖 SM 12.1 Marlin 自动检测

**izhuhaoran 动态:**
- 5 open PRs，无新 merge
- 专注 Model Runner V2、HMA、Blackwell 方向
- 节奏稳定，非批量提交

**本地环境:**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- 分支：fix/helm-chart-selector-labels
- 工作区：干净

**待办事项:**
1. Slack #pr-reviews 求助 PR #37959 加 `ready` label
2. 等 #38126 merge 后检查代码
3. 如 #38126 未覆盖 SM 12.1 Marlin 检测 → 准备 #37141 PR

---

## 2026-03-26 19:09 更新

**PR #38126 状态：** 仍未 merge，最后更新 ~22h 前
- 等待 NVIDIA maintainer johnnynunez 推进
- merge 后将检查代码是否覆盖 SM 12.1 Marlin 自动检测
- 如未覆盖，继续推进 #37141

**PR #37959 状态：** 62h 无人类 review
- Slack CLI 不可用，无法在 #pr-reviews 求助
- 选项：在 PR 上追加评论或继续等待（周末 review 可能较慢）

**DGX Spark 环境稳定：**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- 无新问题发现

**下一步计划：**
1. 等 #38126 merge 后检查代码
2. 如未覆盖 Marlin auto-detect → 准备 #37141 PR
3. 继续跟进 #37959 的 review 进展

---

## 2026-03-26 20:13 更新

**PR #38126 状态跟踪：**
- 仍 open，最后更新 ~23h 前
- 等待 merge 后检查代码变更
- 关键问题：是否覆盖 SM 12.1 (GB10) Marlin 自动检测？
- 如未覆盖 → 推进 #37141

**PR #37959 (Helm chart 修复)：**
- 64h 无人类 review
- 仍缺 `ready` label
- 今天不追加评论，继续等待

**DGX Spark 环境稳定：**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- CUDA: 13.0
- 无新问题发现

---

## 2026-03-26 22:23 更新

**PR #38126 状态：** 仍 open，最后更新 ~25h 前
- 等待 NVIDIA maintainer johnnynunez 推进
- merge 后将检查代码是否覆盖 SM 12.1 Marlin 自动检测
- 如未覆盖，继续推进 #37141

**PR #37959 状态：** 68h 无人类 review
- 周末 review 节奏慢是正常现象
- 周一 (3 月 30 日) 如仍无进展，考虑追加评论

**DGX Spark 环境稳定：**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- CUDA: 13.0
- 无新问题发现

**今日产出：**
- 0 新 PR（符合每天≤1 上限）
- 持续跟进现有 PR 状态
- 已准备好 #37141 PR 代码（待 #38126 merge 后决定提交时机）

---

## 2026-03-27 Friday — 07:06 更新

**PR #38126 状态：** 仍 open，**活跃讨论中** 🔥
- 最后更新：~10h 前 (2026-03-26T20:52:26Z)
- 评论数：14 条（活跃讨论）
- 作者：johnnynunez (NVIDIA maintainer)
- Labels: `ready`, `ci/build`, `nvidia`
- 关键进展：讨论持续进行中，接近 merge 状态
- 策略：继续等待 merge
  - merge 后立即检查代码是否覆盖 SM 12.1 Marlin 自动检测
  - 如未覆盖 → 推进 #37141 (Marlin auto-detect for GB10)

**PR #37959 (Helm chart 修复)：**
- 状态：Open ~74h
- 最后更新：~24h 前
- Labels: `bug`, `documentation`（仍缺 `ready`）
- 人类 review: 0
- 决策：周末 review 慢属正常，周一 (3/30) 如仍无进展再追加评论

**DGX Spark 环境稳定：**
- GPU: NVIDIA GB10 (SM 12.1)
- Driver: 580.95.05
- CUDA: 13.0
- 无新问题发现

**今日产出：**
- 0 新 PR（符合每天≤1 上限）
- 持续跟进现有 PR 状态
- PR #38126 讨论活跃（14 条评论），预计近期 merge

**下周计划 (2026-03-27 起)：**
1. **继续观察 #38126** — 如 merge 立即检查代码变更（重点关注 Marlin auto-detect for SM 12.1）
2. **PR #37959** — 周一 (3/30) 如仍无进展，追加简短评论或 Slack 求助
3. **DGX Spark 测试** — 记录 baseline 性能数据（Marlin vs CUTLASS）
4. **社区参与** — 评论 1-2 个相关 issue（学习性质，DGX/KVCache 方向）
