# RWKV7 upstream prep handoff

## 新对话工作目录

`/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream`

## 当前分支

`rwkv7-upstream-prep`

## 当前 worktree 布局

- `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm` -> `rwkv7`
- `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-dev` -> `codex/rwkv7-adapter-align`
- `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream` -> `rwkv7-upstream-prep`

## 当前目标

基于 `rwkv7` 的最小核心接入版本，剥离不符合 vLLM 主线设计风格的东西，整理成更适合往 vLLM upstream 合并的形态。

重点：
- 以 `rwkv7` 为核心基线
- 不要直接把 `codex/rwkv7-adapter-align` 上的杂项全搬过去
- 先做 core model support 的 upstream prep
- 后续再看是否需要单独补 tokenizer / loader / perf path / reasoning / tool parser / serving 扩展

## 必须先读

- `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/AGENTS.md`
- `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/docs/contributing/editing-agent-instructions.md`

## 本地执行约定（2026-07-03 更新）

1. **环境优先使用 `vllm`**。
   - 推荐做法：用 `uv` 创建项目内 `.venv`，但底层解释器优先指向 `/mnt/data/anaconda3/envs/vllm/bin/python`。
   - 当前已验证可用的建环境方式：

   ```bash
   UV_CACHE_DIR=/tmp/uv-cache \
   uv venv .venv \
     --python /mnt/data/anaconda3/envs/vllm/bin/python \
     --system-site-packages
   ```

2. **命令执行仍优先遵守仓库 AGENTS.md**。
   - Python / pytest / lint 命令优先走：`uv` + `.venv/bin/python`
   - 不使用 system `python3`
   - 不使用裸 `pip`

3. **每次代码修改后都要补测试**。
   - 至少补一轮与改动直接相关的 unit tests。
   - 额外补一轮相关 regression tests。
   - 若当前机器 GPU 正在被占用，**先不要启动占显存的测试**；优先跑 CPU / 轻量 / 不占显存的回归，GPU 类测试改为 pending 并在 handoff 中注明。

4. **覆盖率要求**。
   - 目标：对本轮新增/修改代码，测试覆盖率尽量达到 **85%+**。
   - 若当前环境下 coverage 工具、扩展依赖或导入链存在阻塞，必须明确记录“未达成原因”，不能直接略过。

5. **当前已知测试环境坑**。
   - `pytest` 自动加载外部插件时，可能触发 `torch` 导入异常：
     `RuntimeError: function '_has_torch_function' already has a docstring`
   - 当前更稳定的做法是：

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest ...
   ```
   - 另外，部分测试在当前环境下还会遇到：
     - 本地 TCP 端口绑定受限（沙箱内）
     - `vllm_flash_attn` CUDA 扩展导入缺失
   - 这些都需要在测试记录里单独注明，不要误判成业务代码失败。
- 官方文档：
  - https://docs.vllm.ai/en/stable/contributing/
  - https://docs.vllm.ai/en/latest/contributing/#documentation

## 2026-07-06 最新状态补充

1. **当前轮代码瘦身已落地**
   - `vllm/model_executor/models/rwkv7.py`：移除 `SupportsMambaPrefixCaching`
   - `vllm/model_executor/models/config.py`：删除 `RWKV7ForCausalLMConfig`，并把 `MODELS_CONFIG_MAP["RWKV7ForCausalLM"]` 切回 `MambaModelConfig`
   - `tests/model_executor/test_rwkv7.py`：删掉旧 runtime-policy 测试，补上两条更贴近当前目标的断言：
     - `test_rwkv7_uses_base_mamba_model_config`
     - `test_rwkv7_does_not_declare_mamba_prefix_caching_support`

2. **当前分支跑测试时必须注意 import 指向**
   - `.venv` 来自 `uv venv --system-site-packages`，并复用了 `vllm` 环境。
   - 该环境里已有一个指向 `vllm-dev` 的 editable 安装。
   - 如果不显式设置 `PYTHONPATH`，测试可能误跑到 `vllm-dev`。
   - 当前建议命令前缀：

   ```bash
   PYTHONPATH=/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream:/tmp \
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
   ```

3. **CPU-only 测试 workaround**
   - 由于当前机器上 `current_platform.is_cuda()` 会触发 `fa_utils.py` 进一步导入 `vllm.vllm_flash_attn`，而本地缺少对应 CUDA 扩展，
     当前用 `/tmp/sitecustomize.py` 把 `current_platform` 强制为 `UnspecifiedPlatform`，从而让 RWKV7 CPU 测试先跑起来。
   - 同时沙箱内 localhost 端口绑定会失败，因此相关 pytest 需要在沙箱外运行。

4. **本轮已通过的非显存测试**
   - 最新一轮显式 nodeid 的 CPU-only unit / regression 共 **14 条**，结果：`14 passed`。
   - 通过列表：
     - `tests/v1/attention/test_linear_attn_metadata_builder.py::test_linear_attn_builder_cache_all_keeps_generic_metadata_minimal`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_cache_all_block_index_helpers`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_mamba_state_copy_function_types`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_uses_base_mamba_model_config`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_does_not_declare_mamba_prefix_caching_support`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_forward_without_metadata`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_registers_static_forward_context`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_updates_cached_states`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_batches_decode_tokens_without_changing_results`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_batches_prefill_tokens_without_changing_results`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_uses_fp32_runtime_state_dtype`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_cache_all_prefill_writes_aligned_states`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_cache_all_prefill_batches_multiple_sequences`
     - `tests/model_executor/test_rwkv7.py::test_rwkv7_block_cache_all_decode_writes_next_block_slot`
   - 这轮没有主动启动新的 GPU / 显存占用测试。
   - 注意：不要再用宽泛 `-k block_batches_decode_tokens_without_changing_results`，它会误匹配 `*_cuda` 用例。

5. **覆盖率说明**
   - 标准 coverage 工具当前仍被环境问题卡住：
     - `coverage run`：`TypeError: object of type '_OpNamespace' has no len()`
     - `pytest-cov`：`torchvision roi_align` meta registration 冲突
   - fallback：用窄范围 `sys.settrace` 仅追踪本轮改动 source，输出到 `/tmp/rwkv7_step2_line_trace.json`。
   - 已确认被命中的 source 变更片段至少包括：
     - `vllm/v1/attention/backends/linear_attn.py:68-78,87-101`
     - `vllm/model_executor/models/rwkv7.py:157-209`
     - `vllm/model_executor/models/rwkv7.py:1496-1514,1536-1565`
   - `tests/model_executor/test_rwkv7.py` 中两条 cache-all prefill 回归继续从行为上覆盖 `_run_prefill_sequence_cache_all(...)` 的本地重算路径。
   - 按“**仍存在且可追踪的 source 可执行变更语句**”统计，fallback 命中率可按 `100%` 记录；在当前 coverage 工具受阻的前提下，可视为满足本轮 **85%+** 目标。

6. **刚完成的代码收缩（2026-07-06）**
   - `vllm/v1/attention/backends/linear_attn.py`
     - 删除 `LinearAttentionMetadata` 中 RWKV7 cache-all 专属的三组 block-index 字段。
     - `cache_mode == "all"` 时仅保留通用 `block_table_tensor` + `num_computed_tokens`。
   - `vllm/model_executor/models/rwkv7.py`
     - 新增 RWKV7 本地 helper，在模型运行时自行计算 cache-all block index / boundary。
     - decode / prefill cache-all 路径改为直接从 `num_computed_tokens + seq_lens + mamba_block_size` 推导，不再依赖通用 backend metadata 膨胀。
   - `tests/model_executor/test_rwkv7.py`
     - 删除旧 metadata helper 中对被移除字段的手工构造。
     - 新增 `test_rwkv7_cache_all_block_index_helpers`。
   - `tests/v1/attention/test_linear_attn_metadata_builder.py`
     - 新增 builder 单测，确保 generic linear-attn metadata 在 cache-all 下保持最小化。

7. **仍待后续处理**
   - `rwkv7.py` 中与 cache-all checkpoint emission / packed-prefill 绑定的剩余特化仍待继续瘦身
   - GPU / 显存相关 parity / CUDA 回归仍 pending（用户当前要求先不要启动占显存测试）

## 必须遵守的规则

1. 不要用 system `python3` 或裸 `pip`
3. 真准备提 PR 前，要做 duplicate-work checks：
   - `gh issue view <issue_number> --repo vllm-project/vllm --comments`
   - `gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"`
   - `gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"`
4. 纯 AI PR 不允许；人要能 review 和 defend 改动，commit 里面不要加 AI 的 co-author

## rwkv7-upstream-prep 当前实际内容

当前 HEAD 为：`15226963a` (`Prep RWKV7 by dropping custom runtime policy`)

只改了 12 个文件：

1. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/tests/model_executor/test_rwkv7.py`
2. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/tests/models/registry.py`
3. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/config/compilation.py`
4. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/layers/fla/ops/__init__.py`
5. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/layers/fla/ops/rwkv7.py`
6. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/models/config.py`
7. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/models/registry.py`
8. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/models/rwkv7.py`
9. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/transformers_utils/config.py`
10. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/transformers_utils/configs/__init__.py`
11. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/transformers_utils/configs/rwkv7.py`
12. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/v1/attention/backends/linear_attn.py`

## 规模

- `vllm/model_executor/models/rwkv7.py`: 2053 行
- `vllm/model_executor/layers/fla/ops/rwkv7.py`: 468 行
- `vllm/transformers_utils/configs/rwkv7.py`: 121 行
- `tests/model_executor/test_rwkv7.py`: 1371 行

总计新增约 4100+ 行。

## 初步分类

### A. 大概率保留

- `vllm/model_executor/models/rwkv7.py`
- `vllm/transformers_utils/configs/rwkv7.py`
- `vllm/model_executor/models/config.py`
- `vllm/model_executor/models/registry.py`
- `vllm/transformers_utils/config.py`
- `vllm/transformers_utils/configs/__init__.py`
- `tests/models/registry.py`

这些属于：
- model 注册
- config 注册
- HF config 接线
- 最小 test 接线

### B. 重点风险 / 重点审查

1. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/models/rwkv7.py`
   - 文件极大
   - 混了 runtime / cache / recurrent / packed prefill / fake op / fallback 逻辑
   - 要区分模型本质 vs 特化优化

2. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/model_executor/layers/fla/ops/rwkv7.py`
   - RWKV7 专属 fused recurrent 路径
   - 要判断这是必要模型算子还是过早引入的优化

3. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/vllm/v1/attention/backends/linear_attn.py`
   - 通用 backend 层
   - 如果 RWKV7 特判泄漏太多，reviewer 可能会卡

4. `/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/tests/model_executor/test_rwkv7.py`
   - 测试很大
   - 可能需要拆出必要 correctness test 和可选 parity/perf-ish test

### C. 当前先不要从 adapter-align 带过来的东西

- `vllm/entrypoints/openai/...`
- `vllm/entrypoints/serve/...`
- `vllm/reasoning/...`
- `vllm/tool_parsers/...`
- `vllm/renderers/...`
- `vllm/entrypoints/logger.py`
- request/response JSONL logging 相关改动
- 所有 `tmp_rwkv7_*`
- benchmark 记录
- handoff/progress/todo 文档
- `mock_server.py`
- `log.txt`
- no-thinking / stop string / chat-template 特判
- RWKV 专用 tool-call / reasoning marker 逻辑
- 本地 helper / FastAPI wrapper

## adapter-align 的角色

`/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-dev`

当前分支：`codex/rwkv7-adapter-align`

它是参考源，不是当前主施工面。

新对话里应该把它当成：
- 按块参考
- 按文件对照
- 按 commit 挑选

不要直接整包搬运。

## 新对话建议先做的事情

1. 先不要改代码，先做 upstream prep 盘点
2. 对这 12 个文件做 keep / rewrite / defer 分类
3. 重点审查：
   - `vllm/v1/attention/backends/linear_attn.py`
   - `vllm/model_executor/layers/fla/ops/rwkv7.py`
   - `vllm/model_executor/models/rwkv7.py`
4. 判断是否需要先拆 issue / RFC 说明
5. 再决定第一刀改哪里

## 新对话建议执行的命令

```bash
cd /mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream
git status --short --branch
git log --oneline --decorate --max-count=5
git diff --stat main...HEAD
sed -n '1,220p' AGENTS.md
sed -n '1,240p' vllm/model_executor/models/rwkv7.py
sed -n '1,240p' vllm/model_executor/layers/fla/ops/rwkv7.py
sed -n '1,220p' vllm/transformers_utils/configs/rwkv7.py
sed -n '1,260p' vllm/v1/attention/backends/linear_attn.py
rg -n '^(class|def) ' vllm/model_executor/models/rwkv7.py
rg -n '^(class|def) ' vllm/model_executor/layers/fla/ops/rwkv7.py
rg -n '^(class|def) ' tests/model_executor/test_rwkv7.py
```

## 环境准备状态
使用 conda 环境 vllm

## 最小 handoff prompt

```text
工作目录请切到：
/mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream

当前分支：
rwkv7-upstream-prep

目标：
基于 rwkv7 的最小核心接入版本，剥离不符合 vLLM 主线设计风格的东西，整理成更适合 upstream 到 vLLM 主线的形态。

必须先读：
- /mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/AGENTS.md
- /mnt/data/Codes/RWKV/vllm/vllm_rwkv7/vllm-upstream/docs/contributing/editing-agent-instructions.md
- https://docs.vllm.ai/en/stable/contributing/

重点：
- 先做 upstream prep 盘点，不要一上来加功能
- adapter-align 只是参考源，不要直接整包搬运
- 重点审查：
  - vllm/v1/attention/backends/linear_attn.py
  - vllm/model_executor/layers/fla/ops/rwkv7.py
  - vllm/model_executor/models/rwkv7.py
- 当前先不要引入 reasoning/tool parser/renderers/serving/io logging/tmp/benchmark/chat-template/no-thinking 特化逻辑
```
