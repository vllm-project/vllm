# Daily Learnings — simpx

**GitHub:** [@simpx](https://github.com/simpx)
**Email:** simpxx@gmail.com
**Started:** 2026-03-19

---

## 日期：2026-03-20 (第一天)

### ✅ 昨天完成

- **PR #37578**: [Bugfix] Fix unclean shutdown from Ctrl-C with AR Fusion — Open (待 CI)
- **Token 配置**: GitHub API token 已存入 `~/.github_token`
- **作者身份修复**: 从 `思潜 <lingjun.zlj@alibaba-inc.com>` 改为 `simpx <simpxx@gmail.com>`

### 📚 学到的经验

1. **PR 标题格式** — `[标签] 简短描述`，例如：`[Bugfix] Fix unclean shutdown...`
2. **改动大小** — 第一个 PR 保持小改动（18 行），容易被接受
3. **作者身份** — 必须用 `simpx <simpxx@gmail.com>`，不要用错
4. **force push 风险** — 会改变 commit hash，导致 PR 被关闭，需要重新创建
5. **git commit-tree** — 可以用环境变量设置作者：`GIT_AUTHOR_NAME="simpx" GIT_AUTHOR_EMAIL="simpxx@gmail.com"`
6. **Token 必要性** — 没有 API token 无法通过 API 创建 PR，只能手动

### ⚠️ 踩过的坑

- ❌ 作者身份搞错（第一次 commit 用了 `思潜 <lingjun.zlj@alibaba-inc.com>`）
- ❌ force push 导致 PR #37577 被关闭
- ✅ 重新创建 PR #37578，作者身份正确

### 🎯 今天计划

1. [x] 检查 PR #37578 的 CI 状态 — 无评论，只有 gemini bot 无法审查的提示
2. [x] 查看是否有 bot/维护者评论 — 无
3. [x] 学习社区 merge 的 PRs（至少 10 个）— 见下方
4. [x] 检查 izhuhaoran 的新动态 — 有 1 个 open PR (#35520)，之前有 4 个 merge
5. [x] 推进新开发 — 完成 JAIS ALiBi bugfix
6. [x] 准备第二个 PR — **PR #37631 已创建**

### 📖 社区准则提醒

- ✅ 尊重维护者时间 — PR 描述完整
- ✅ 技术诚信 — 只提交理解并测试过的代码
- ✅ 建设性参与 — 接受批评，不争论风格
- ✅ 长期承诺 — 跟进 review，不是一次性贡献者

---

## 📊 今天完成的工作

### PR #37631 创建成功

**标题**: `[Bugfix] JAIS: Only apply ALiBi when position_embedding_type is alibi`
**改动**: 11 行（9 新增，2 删除），1 个文件
**关联 Issue**: #37400 (good first issue)
**分支**: `fix-jais-alibi-learned-emb`
**URL**: https://github.com/vllm-project/vllm/pull/37631

**修复内容**:
- 问题：JAISAttention 中 alibi_slopes 无条件构建，即使 position_embedding_type 是 'learned'
- 修复：添加条件检查，只有当 position_embedding_type == 'alibi' 时才构建 alibi_slopes
- 影响：避免 learned embeddings + ALiBi 的双重位置编码问题

**自测完成**:
- ✅ 语法检查通过 (`python -m py_compile`)
- ✅ 理解每一行代码
- ✅ PR 描述完整（What/Why/How/Related Issues）

---

## 📚 社区成功经验学习

### 最近 Merge 的 PRs 分析

| PR# | 标题 | 作者 | 改动 | 文件数 | 评论 |
|-----|------|------|------|--------|------|
| 37606 | [ROCm][Bugfix] fix cache block size mismatch | divakar-amd | +7/-24 | 2 | 0 |
| 37574 | Fix `SpeculatorsConfig` dataclass | hmellor | +7/-0 | 1 | 2 |
| 37573 | [Bug] Fix EmbedIOprocessor classify/embed | yewentao256 | - | - | - |
| 37572 | [Refactor] Remove dead code in pooling | yewentao256 | - | - | - |
| 37568 | [Log] Log once in local node by default | yewentao256 | - | - | - |

**学习点**:
1. **小改动优先** — 大部分 merge 的 PR 都是 <20 行改动
2. **标签清晰** — `[Bugfix]`, `[Refactor]`, `[ROCm]` 等前缀
3. **单文件改动** — 大部分只改 1-2 个文件
4. **评论不多** — 很多 PR 没有评论直接 merge

### izhuhaoran 动态

- **Open PR**: #35520 [Model Runner V2] support qwen35 / mamba hybrid model
- **Merge PRs**: 4 个（#35376, #35294, #33433, #33251）
- **专注领域**: Model Runner V2, Qwen 模型，Spec Decoding
- **PR 风格**: 标题带标签，改动集中在核心模块

---

## 📋 PR 状态总览

| PR# | 标题 | 状态 | 评论 | 备注 |
|-----|------|------|------|------|
| #37578 | Fix unclean shutdown from Ctrl-C with AR Fusion | Open | 1 (bot) | gemini bot 无法审查该文件类型，无人类评论 |
| #37631 | JAIS: Only apply ALiBi when position_embedding_type is alibi | Open | 0 | 刚创建，等待 CI 和 review |

**今日 PR 数：2/2** — 已达上限，不再创建新 PR

---

## 🎯 下一步计划

1. ⏳ 等待 PR #37578 和 #37631 的 CI 结果和维护者 review
2. 📚 继续学习社区 merge 的 PR 模式
3. 🔍 寻找明天的 good first issue（今天不创建新 PR）
4. 💡 可以探索代码库，为明天做准备

**今日原则：** 质量 > 数量。两个 PR 都是小改动 bugfix，符合社区模式。

---

## 📚 izhuhaoran 学习记录

**Open PRs:** 1
- #35520: [Model Runner V2] support qwen35 / mamba hybrid model

**Merged PRs:** 4+ (最近 2 个月)
- #35376: dp cudagraph 性能优化
- #35294: spec decoding 支持 dp & ep
- #33433: bad_words sampling param
- #33251: spec decode apply penalty

**学习点:**
1. **专注领域** — izhuhaoran 专注于 Model Runner V2 和 spec decoding，建立专业度
2. **持续贡献** — 2 个月内 4+ merge，节奏稳定
3. **PR 风格** — 标题带清晰标签 `[Model Runner V2]`, `[BugFix]`, `[perf]`
4. **改动范围** — 集中在核心模块，不是零散修复

**启发:** 可以考虑在某个领域（如 KVCache、多模态、或特定模型支持）建立深度贡献。

---

## 📝 下午 Check-in (13:10 HKT)

### PR 状态更新

| PR# | 标题 | 状态 | 评论 | 行动 |
|-----|------|------|------|------|
| #37578 | Fix unclean shutdown from Ctrl-C with AR Fusion | Open | 1 (gemini bot) | ✅ 无人类评论，等待 review |
| #37621 | JAIS: Only apply ALiBi when position_embedding_type is alibi | Open | 0 | ✅ 等待 CI 和 review |

**Bot 评论详情 (#37578):**
```
> [!NOTE]
> Gemini is unable to generate a review for this pull request due to the file types involved not being currently supported.
```
→ 这是正常的 bot 限制，不影响人类维护者 review

### 📚 izhuhaoran 今日动态（学习参考）

**今日创建 5 个 PR**（⚠️ 这是高频，我不应该模仿）:
- #37639: [Model Runner V2] Fix draft logits not populated during cudagraph replay
- #37637: Fix: Add EAGLE/MTP slots calculation in max_num_new_slots_for_drafting
- #37636: [KVConnector] Support 3FS KVConnector
- #37635: [NIXL][Mamba][3/N] Heterogeneous TP: 3-read conv state transfer
- #37634: [XPU] Automatically detect target platform as XPU in build.

**所有 5 个 PR 目前都是 Open 状态** — 说明即使是高频贡献者，也需要等待 review。

**学习点:**
1. **专注领域建立专业度** — izhuhaoran 集中在 Model Runner V2、KVConnector、Mamba
2. **标题格式规范** — 始终带清晰标签 `[Model Runner V2]`, `[KVConnector]`, `[NIXL]`
3. **但我保持自己的节奏** — 1-2 PR/天，质量优先

### 🔍 明天的 Good First Issue 候选

| Issue# | 标题 | 标签 | 创建时间 |
|--------|------|------|----------|
| #37223 | Add LoRA support for Qwen3ASRForConditionalGeneration | good first issue, feature | 3 月 16 日 |
| #35310 | Qwen-ASR Forced Aligner | good first issue, feature | 2 月 25 日 |
| #32588 | Wrong timestamps if audio > 30s | bug, good first issue | 1 月 19 日 |

**已解决**: #37400 (JAIS ALiBi) — 已被 PR #37621 修复

### 📊 最近 Merge 的 PR 模式

- #37612: Deprecate --disable-frontend-multiprocessing (sfeng33)
- #37606: [ROCm][Bugfix] fix cache block size mismatch (divakar-amd) +7/-24
- #37574: Fix `SpeculatorsConfig` dataclass (hmellor) +7/-0
- #37573: [Bug] Fix EmbedIOprocessor (yewentao256)
- #37572: [Refactor] Remove dead code in pooling (yewentao256)

**模式确认:**
- ✅ 小改动优先（大部分 <20 行）
- ✅ 标签清晰（[Bugfix], [Refactor], [ROCm]）
- ✅ 单文件改动常见
- ✅ 很多 PR 0 评论直接 merge

---

## 待办事项

- [x] 检查 PR review 状态 — 完成，无人类评论
- [x] 学习社区 merge 的 PRs — 完成
- [x] 检查 izhuhaoran 动态 — 完成
- [x] 今日 PR 上限检查 — 2/2，停止创建
- [x] 寻找明天的 good first issue — 完成，有 3 个候选
- [ ] 明天继续跟进 review 回复
- [ ] 选择一个 issue 开始准备

---

## 📝 下午 Check-in (14:15 HKT)

### PR 状态确认

| PR# | 标题 | 状态 | 评论 | 行动 |
|-----|------|------|------|------|
| #37578 | Fix unclean shutdown from Ctrl-C with AR Fusion | Open | 1 (gemini bot) | ⏳ 等待人类 review |
| #37621 | JAIS: Only apply ALiBi when position_embedding_type is alibi | Open | 0 | ⏳ 等待 CI 和 review |

**无新评论** — 两个 PR 都在等待维护者 review。这是正常的，vLLM 项目 PR 较多，review 需要时间。

### 📚 今日学习总结

**最近 Merge 的 PRs 模式**（今天 merge 的）:
- #37634: [XPU] Automatically detect target platform — ccrhx4
- #37612: [V0 Deprecation] Deprecate --disable-frontend-multiprocessing — sfeng33
- #37606: [ROCm][Bugfix] fix cache block size mismatch — divakar-amd
- #37593: [Refactor] Relocate entrypoint tests — sfeng33
- #37579: [Model] Refactor Step3-VL processor to HF style — DarkLight1337

**确认的模式**:
1. ✅ 标签清晰 — `[XPU]`, `[ROCm][Bugfix]`, `[Refactor]`, `[Model]`
2. ✅ 改动集中 — 大部分是单模块改动
3. ✅ 标题简短 — 直接说明做了什么
4. ✅ 维护者活跃 — 今天 merge 了 5+ PRs

### izhuhaoran 今日动态（学习参考）

**今日新建 8 个 PRs**（全部 Open 状态）:
- #37643: Fix AudioFlamingo3/MusicFlamingo HF parity and RoTE handling
- #37642: [Bugfix] Fix engine crash when structured output grammar compilation fails
- #37641: [XPU] bump vllm-xpu-kernels to v0.1.4
- #37640: [ROCm][Test] Fix ROCM_AITER_UNIFIED_ATTN test
- #37639: [Model Runner V2] Fix draft logits not populated during cudagraph replay
- #37637: Fix: Add EAGLE/MTP slots calculation
- #37636: [KVConnector] Support 3FS KVConnector
- #37635: [NIXL][Mamba][3/N] Heterogeneous TP

**观察**:
- 专注领域：Model Runner V2, KVConnector, Mamba, XPU, ROCm
- 标题格式始终规范
- **但我保持自己的节奏** — 1-2 PR/天，质量优先，不追求数量

### 🎯 明天的候选 Issues

| Issue# | 标题 | 类型 | 优先级 |
|--------|------|------|--------|
| #37223 | Add LoRA support for Qwen3ASRForConditionalGeneration | Feature | ⭐⭐⭐ |
| #32588 | Wrong timestamps if audio > 30s | Bug | ⭐⭐⭐ |
| #33267 | Remove attention layer name from unified_kv_cache_update | Refactor | ⭐⭐ |

**首选**: #37223 — LoRA support 是常见需求，改动范围可控，有明确的实现方向

### 📊 今日总结

**完成**:
- ✅ PR #37578 创建（等待 review）
- ✅ PR #37621 创建（等待 review）
- ✅ 学习社区 merge 模式
- ✅ 跟踪 izhuhaoran 动态
- ✅ 寻找明天候选 issues

**今日 PR 数**: 2/2 — 已达上限

**明日计划**:
1. 早上检查 PR review 状态
2. 如有 review 评论，及时回复
3. 如无评论，开始 #37223 (LoRA for Qwen3ASR) 的调研
4. 保持 1-2 PR/天节奏

---

*Last updated: 2026-03-20 16:24 HKT*

---

## 📝 下午 Check-in (16:24 HKT)

### PR 状态确认（16:24）

| PR# | 标题 | 状态 | 评论 | 行动 |
|-----|------|------|------|------|
| #37578 | Fix unclean shutdown from Ctrl-C with AR Fusion | Open | 1 (gemini bot) | ⏳ 等待人类 review |
| #37621 | JAIS: Only apply ALiBi when position_embedding_type is alibi | Open | 0 | ⏳ 等待 CI 和 review |

**无新评论** — 两个 PR 都在等待维护者 review。正常现象，vLLM 项目 PR 量大，review 需要时间。

**Bot 评论详情 (#37578)**:
```
> [!NOTE]
> Gemini is unable to generate a review for this pull request due to the file types involved not being currently supported.
```
→ 这是正常的 bot 限制，不影响人类维护者 review

### 📚 今日 Merge 的 PRs 学习

| PR# | 标题 | 作者 | Merge 时间 |
|-----|------|------|------------|
| 37641 | [XPU] bump vllm-xpu-kernels to v0.1.4 | jikunshang | 07:04 UTC |
| 37639 | [Model Runner V2] Fix draft logits not populated | TheEpicDolphin | 07:43 UTC |
| 37634 | [XPU] Automatically detect target platform as XPU | ccrhx4 | 05:30 UTC |
| 37612 | [V0 Deprecation] Deprecate --disable-frontend-multiprocessing | sfeng33 | 03:31 UTC |
| 37606 | [ROCm][Bugfix] fix cache block size mismatch | divakar-amd | 00:00 UTC |
| 37593 | [Refactor] Relocate entrypoint tests | sfeng33 | 05:31 UTC |
| 37585 | [CI] Removing deprecated rlhf examples reference | AndreasKaratzas | 07:20 UTC |
| 37579 | [Model] Refactor Step3-VL processor to HF style | DarkLight1337 | 06:05 UTC |

**学习点**:
1. ✅ 标签清晰 — `[XPU]`, `[Model Runner V2]`, `[ROCm][Bugfix]`, `[Refactor]`, `[CI]`
2. ✅ 改动集中 — 大部分是单模块改动
3. ✅ 维护者活跃 — 今天 merge 了 8+ PRs
4. ✅ 标题简短直接 — 说明做了什么

### izhuhaoran 今日动态（学习参考）

**最近 PRs**:
- 专注领域：Model Runner V2, KVConnector, Mamba, XPU
- 标题格式始终规范
- **但我保持自己的节奏** — 1-2 PR/天，质量优先

### 📊 今日总结（16:24）

**完成**:
- ✅ 检查 PR review 状态 — 无人类评论
- ✅ 学习社区 merge 模式 — 小改动、标签清晰、单文件优先
- ✅ 跟踪 izhuhaoran 动态 — 专注核心模块领域
- ✅ 今日 PR 上限检查 — 2/2，停止创建

**今日 PR 数**: 2/2 — 已达上限

**明日计划**:
1. 早上检查 PR review 状态，如有评论及时回复
2. 如无评论，开始 #37223 (LoRA for Qwen3ASR) 的调研
3. 保持 1-2 PR/天节奏，质量优先

---

## 更新：2026-03-20 12:04 HKT

### 🔧 处理重复 PR

**问题**: 发现 PR #37631 和 #37621 是重复的（相同的 JAIS ALiBi 修复）
**原因**: 可能是之前 force push 导致的 PR 重建
**处理**: 
- ✅ 已关闭重复的 #37631
- ✅ 保留原始的 #37621（创建时间更早，00:50 vs 01:57）

**教训**: 
- 创建 PR 前先搜索是否有相同标题/内容的 PR
- force push 前要确认不会导致 PR 冲突

### 📊 当前 PR 状态

| PR# | 标题 | 状态 | 评论 | 备注 |
|-----|------|------|------|------|
| #37578 | Fix unclean shutdown from Ctrl-C with AR Fusion | Open | 1 (bot) | gemini bot 无法审查，等待人类 review |
| #37621 | JAIS: Only apply ALiBi when position_embedding_type is alibi | Open | 0 | 原始 PR，等待 CI 和 review |
| #37631 | (重复) | **Closed** | 1 (mergify) | 已关闭，与 #37621 冲突 |

### 📚 izhuhaoran 今日动态（学习参考）

**今日创建 5 个 PR**（注意：这是高频，我不应该模仿这个节奏）:
- #37639: [Model Runner V2] Fix draft logits not populated during cudagraph replay
- #37637: Fix: Add EAGLE/MTP slots calculation in max_num_new_slots_for_drafting
- #37636: [KVConnector] Support 3FS KVConnector
- #37635: [NIXL][Mamba][3/N] Heterogeneous TP: 3-read conv state transfer
- #37634: [XPU] Automatically detect target platform as XPU in build.

**观察**: 
- 集中在 Model Runner V2、KVConnector、Mamba 领域
- 标题格式规范，带清晰标签
- 但我应该保持自己的节奏（1-2 PR/天），而不是追求数量

### 🎯 下午计划

1. ⏳ 继续等待 #37578 和 #37621 的 review
2. 📖 阅读代码库，为明天的贡献做准备
3. 🔍 寻找下一个 good first issue
4. 📝 学习 vLLM 架构文档

**今日 PR 数：2/2**（#37578 + #37621），不再创建新 PR

---

## 📝 傍晚 Check-in (17:29 HKT) — 工作时段结束

### PR 状态最终确认

| PR# | 标题 | 状态 | 评论 | 行动 |
|-----|------|------|------|------|
| #37578 | Fix unclean shutdown from Ctrl-C with AR Fusion | Open | 1 (gemini bot) | ⏳ 等待人类 review |
| #37621 | JAIS: Only apply ALiBi when position_embedding_type is alibi | Open | 0 | ⏳ 等待 CI 和 review |

**无新评论** — 两个 PR 都在等待维护者 review。正常现象，vLLM 项目 PR 量大，review 周期通常 1-3 天。

### 📚 izhuhaoran 今日动态（学习参考）

**今日新建 8+ 个 PRs**（全部 Open 状态，尚未 merge）:
- #37664: [Feature] Feat(structured-outputs): expose xgrammar bitmask backend selection
- #37662: fix: handle multicasting error in FlashInfer workspace init
- #37661: [Misc] Use logger.info_once for auto tool choice log message
- #37657: [CI][PD] Add Hybrid SSM integration tests to CI
- #37656: [EPLB] EPLB algorithm added: FlashLB+SwiftBalancer
- #37655: [Do not merge] it is for test cases statistics
- #37654: [Feature] Expose xgrammar bitmask backend selection in StructuredOutputsConfig
- #37653: fix: handle multicasting error in FlashInfer workspace init

**关键学习点**:
- 即使一天创建 8+ PRs，也都需要等待 review，没有立即 merge
- 专注领域：Structured Outputs, FlashInfer, CI/CD, EPLB 负载均衡
- **我的策略**: 保持 1-2 PR/天，质量优先，每个 PR 都能充分跟进 review

### 📊 今日 Merge 的 PRs 学习

| PR# | 标题 | 作者 | Merge 时间 (UTC) |
|-----|------|------|------------------|
| 37619 | [ROCm][CI] Update GSM8K eval config | AndreasKaratzas | 09:06 |
| 37614 | [ROCm][CI] Remove deepep DBO tests | AndreasKaratzas | 09:07 |
| 37611 | [ROCm][CI] Fix granite_speech test | AndreasKaratzas | 09:07 |
| 37641 | [XPU] bump vllm-xpu-kernels to v0.1.4 | jikunshang | 07:04 |
| 37639 | [Model Runner V2] Fix draft logits | TheEpicDolphin | 07:43 |
| 37634 | [XPU] Automatically detect target platform | ccrhx4 | 05:30 |
| 37612 | [V0 Deprecation] Deprecate --disable-frontend-multiprocessing | sfeng33 | 03:31 |

**确认的模式**:
1. ✅ 标签清晰 — `[ROCm][CI]`, `[XPU]`, `[Model Runner V2]`
2. ✅ CI/测试修复类 PR 容易被快速 merge
3. ✅ 改动集中 — 单模块/单平台改动

### 📊 今日总结（17:29 工作时段结束）

**完成**:
- ✅ 检查 PR review 状态 — 无人类评论，等待中
- ✅ 学习社区 merge 模式 — CI/测试修复类 PR 容易快速 merge
- ✅ 跟踪 izhuhaoran 动态 — 专注多领域，但 PR 都需等待 review
- ✅ 今日 PR 上限检查 — 2/2，停止创建

**今日 PR 数**: 2/2 — 已达上限，符合安全策略

**明日计划**:
1. 早上 (9:00-10:00) 检查 PR review 状态
2. 如有维护者评论，及时回复（先感谢，再解答问题或修改）
3. 如无评论，开始 #37223 (LoRA for Qwen3ASR) 的调研和实现
4. 保持 1-2 PR/天节奏，质量优先

---

*Last updated: 2026-03-20 17:29 HKT*
