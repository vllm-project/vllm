# MiniCPM-RobotTrack → vLLM 集成 TODO

> 目标:把 `MiniCPM-RobotTrack`(VLA 轨迹策略模型)作为 **in-tree pooling 模型**接入本 vLLM 分支。
> 路线已定:**方案 A(多模态输入通路)** + 复用 `MiniCPMModel` 主干 + 自定义 Pooler 输出 24 维回归向量。
> 本文档汇总了分析结论、已锁定决策、待确认项、分阶段实现清单、正确性陷阱与模板参考。

---

## ✅ 实现状态(2026-07-20:已完成并验证)

**已落地文件**
- `vllm/transformers_utils/configs/minicpm_robottrack.py` — `MiniCPMRobotTrackConfig`。
- `vllm/model_executor/models/minicpm_robottrack.py` — 模型 + MM processor/info/dummy/parser + 自定义轨迹 Pooler。
- 注册:`transformers_utils/config.py` `_CONFIG_REGISTRY`、`configs/__init__.py`、`model_executor/models/registry.py`(`_EMBEDDING_MODELS` 多模态段)、`tests/models/registry.py`。
- 测试:`tests/models/multimodal/pooling/test_minicpm_robottrack.py`。
- Demo:`examples/pooling/robottrack_minicpm_offline.py`(离线端到端,`python examples/pooling/robottrack_minicpm_offline.py --model openbmb/MiniCPM-RobotTrack`;用合成特征跑通 pooling 通路,真实特征替换 `build_dummy_visual_features` 即可)。
- Demo(真·图片输入):`examples/pooling/robottrack_minicpm_video.py`——按帧读图 → **DINOv3+SigLIP 编码(1536 维融合)** → 维护 **32 帧滚动 cache(31 coarse history + 1 fine current)** → vLLM 逐帧推理 → 8 路点 **BEV 叠加画到图上** → 存 `output/`。已在 `track-image/0`(139 帧)实跑通过。

**端到端对拍结论**:vLLM 24 维输出 vs HF golden(在隔离的 `transformers==4.56.2` 环境生成)`torch.allclose(rtol=2e-3, atol=2e-3)` 通过,`max abs diff ≈ 5.9e-4`;batch=2 逐项切分正确、无串扰。ruff + mypy 通过。

**关键落地要点**
- 占位符 token 用**词表内保留 id(`vocab_size-1`)**,不能用 OOV(输入校验会拒绝 > `max(tokenizer.max_token_id, vocab-1)` 的 id);视觉块经 `PromptInsertion(target=end())` 追加到指令后(本模型无 HF processor,`_call_hf_processor` 只做文本分词)。
- 主干 config 需在构造后注入 `rope_theta=10000.0` 到 `rope_parameters`,否则 vLLM LongRoPE 的 base 为 None。
- `get_text_config()` 返回 backbone,让 vLLM 用主干配置规划 KV cache。
- ⚠️ HF 远程代码仅兼容 `transformers>=4.56,<5`;本仓库的 transformers 5.14 会让远程前向出 NaN / 导入失败。对拍 golden 必须用隔离的 4.56.2 环境生成(本仓库 in-tree 实现不依赖 `trust_remote_code`,在 5.14 上正常)。

---

## ✅.1 HF 远程代码升级 transformers 所需改动(上游 OpenBMB 仓库)

> 结论:**在 `transformers>=4.56,<5` 范围内,HF 仓库(`~/model/MiniCPM-RobotTrack/`)代码可原样使用,无需改**。若要支持 **transformers 5.x**,不是放开版本号就行,需要改远程模型代码;下面三处按撞到的先后排列。
>
> 说明:这属于**上游模型作者的事,不阻塞本 vLLM 集成**——我们的 in-tree 实现不加载远程代码,在 5.14 正常;且 4.56.2 生成的 golden 就是模型声明版本(`config.json: transformers_version=4.56.2`)下的**权威参考**。

| # | 症状 / 报错 | 位置 | 性质 | 需要的改动 |
|---|---|---|---|---|
| 1 | `ImportError: cannot import name 'is_torch_fx_available'` | `modeling_minicpm.py:48` `from transformers.utils.import_utils import is_torch_fx_available` | 5.x 删除了该符号 | 删除该导入及其 `torch.fx.wrap` 分支,或 `try/except` 兜底(FX 图叶子标注在推理无用) |
| 2 | `AttributeError: 'MiniCPMRobotTrackModel' object has no attribute 'all_tied_weights_keys'` | 5.x `from_pretrained` 收尾 `_move_missing_keys_from_meta_to_device` 读 `self.all_tied_weights_keys` | 5.x 新的权重绑定机制(`tie_weights()` 里设置),4.56-era 远程 `PreTrainedModel` 未对齐 | 让远程 `PreTrainedModel` 走 5.x 的 `tie_weights`/`_tied_weights_keys` 约定;或声明无绑定权重使基类默认 `{}` 生效 |
| 3 | **前向输出 NaN**(即使 shim 掉 #1/#2 后仍出现) | 远程 MiniCPM 主干前向:真实结构化输入在 `seq_len≈230` 时 `backbone(inputs_embeds=…)` → NaN;**相同长度随机 embeds 却有限**,输入序列本身有限 | 已弃用的 `AttentionMaskConverter`(`_prepare_4d_causal_attention_mask*`)掩码路径 / 远程手写 longrope 与 5.x rope 迁移交互(**未精确锁到具体行,系推断**) | 迁移到 5.x 的 `transformers.masking_utils` 新掩码 API;longrope 改走 5.x 标准 rope。**这才是主要工作量** |

- #1/#2 是导入/加载层,shim 即可绕过;**#3 是数值问题,绕不过**,所以不能在 5.14 里生成可信 golden。
- 若上游要推进 5.x 支持,可先把 #3 精确根因(二分序列长度 + 逐层 hook 定位首个发散点)整理成 HF issue/patch。

**复现所用环境(两侧同输入 `seed=1234` 特征、fp32)**

| 产物 | 环境 | transformers / torch | 生成方式 |
|---|---|---|---|
| vLLM 24 维输出 | 本仓库主 `.venv` | 5.14.1 / 2.11.0+cu130 | 真实 in-tree 模型走 vLLM 引擎 |
| HF golden(参考轨迹) | 隔离 venv `/tmp/hfg` | 4.56.2 / 复用主 `.venv` 的 torch | `uv pip install --python /tmp/hfg/bin/python transformers==4.56.2`;脚本内 `sys.path.append(".venv/.../site-packages")` 借用 torch |

对拍结果:`allclose(rtol=2e-3, atol=2e-3)` 通过,`max abs diff≈5.9e-4`。

---

## 0. 模型速览(为什么不是 drop-in)

- HF 顶层类 `MiniCPMRobotTrackModel`,`architectures=["MiniCPMRobotTrackModel"]`,`model_type="minicpm_robottrack"`。
- 结构 = **MiniCPM4-0.5B 裸 decoder 主干**(无 LM head)+ 输入适配 + 输出回归头:
  - `VisionProjector`:`LayerNorm → Linear(1536→1024) → GELU → Linear(1024→1024)`,投影外部预计算的 **DINOv3+SigLIP 融合特征(1536 维)**。
  - `TemporalMarkerEncoder`:time / stream / camera 三个 Embedding,**在每段相同 time_index 前插一个 marker token**。
  - `control_query`:可学习 `[1,1,1024]` 参数,拼在序列末尾。
  - `FunnelTrajectoryHead`:6 层 MLP `1024→4096→1024→512→256→128→24` + `tanh`,再乘 `output_scale`(xy 维 ×`xy_scale=2.0`)→ reshape `[8,3]`。
- 前向:拼 `[text_emb, history_vis, current_vis, control_query]` → MiniCPM 主干(**causal、`use_cache=False`、单次前向**)→ 取**最后一位(control_query)**hidden → 轨迹头 → `[B,8,3]`。
- **非生成式**:无词表投影、无采样、无自回归 → 属于 vLLM 的 **pooling/回归**范式,不是 `generate`。
- 关键 config:`vision_feature_dim=1536, history_frames=31, coarse_tokens_per_frame=4, fine_tokens_current_frame=64, num_waypoints=8, action_dim=3, max_text_tokens=128, xy_scale=2.0, use_tanh_actions=true`;主干 `hidden=1024, layers=24, longrope, scale_emb=12, dim_model_base=256, scale_depth=1.4`。

---

## 1. 已锁定决策

- [x] **In-tree** 注册(非插件),新增 `vllm/model_executor/models/minicpm_robottrack.py`。
- [x] **视觉特征入模 = 方案 A(多模态通路)**:模型自包含,projector/marker/control_query/head/backbone 全在模型内、由 `load_weights` 加载。
- [x] **主干复用** vLLM 现有 `MiniCPMModel`(`minicpm.py`),通过 `vllm_config.with_hf_config(backbone_cfg)` 传子配置。
- [x] **输出 = 自定义 Pooler**,LAST 池化取 control_query 位 → 24 维;advertise `"embed"` 任务;客户端读 `outputs[0].outputs.data` reshape `[8,3]`。
- [x] **head 层用 `nn.Linear`**(24 维小头,TP 无收益)。
- [x] fp32 精度问题(projector/marker/head)**记录待模型侧确认**;对拍阶段全 fp32。

---

## 2. 架构映射(HF → vLLM)

| HF 模块 | vLLM 落点 | 说明 |
|---|---|---|
| `backbone` (`MiniCPMModel`) | 复用 `minicpm.py: MiniCPMModel` | `with_hf_config(backbone_cfg)` |
| `vision_projector` | 移植为 `nn.Module` | 输入 MM 特征 `[N,1536]` |
| `temporal_markers` | 移植 `nn.Embedding×3` | 在 `get_multimodal_embeddings` 内插 marker |
| `control_query` | `nn.Parameter`,填到末尾占位位 | 保证是序列 LAST |
| `trajectory_head` | 移植(`nn.Linear`+GELU+tanh+scale) | 在 forward 或自定义 Pooler 内 |
| `output_scale` buffer | 按 `config.xy_scale` **重算**,不依赖加载 | |
| forward 读 last hidden | 自定义 Pooler `LastPool` | |
| 输出 `[B,8,3]` | 24 维向量 → 客户端 reshape | |

---

## 3. 分阶段实现清单

### Phase 0 — 脚手架 / config / 注册
- [x] 新增 `vllm/transformers_utils/configs/minicpm_robottrack.py`:`MiniCPMRobotTrackConfig`,`__init__` 把 `backbone_config` **dict 包成真正的 config 对象**。
      → **实际**:`minicpm` 不在 `transformers.CONFIG_MAPPING`,改用 `PretrainedConfig(**bc)`(transformers 5.x 会自动 `rope_scaling→rope_parameters`);**并在构造后注入 `rope_theta=10000.0` 到 `rope_parameters`**,否则 vLLM LongRoPE 的 base 为 None;`get_text_config()` 返回 backbone。
- [x] 在 `vllm/transformers_utils/config.py` 的 `_CONFIG_REGISTRY` 注册 `"minicpm_robottrack" → MiniCPMRobotTrackConfig`(免 `trust_remote_code`),并在 `configs/__init__.py` 导出。
- [x] 在 `vllm/model_executor/models/registry.py` 的 `_EMBEDDING_MODELS`(多模态段)加 `"MiniCPMRobotTrackModel": ("minicpm_robottrack", "MiniCPMRobotTrackModel")`。

### Phase 1 — 模型类
- [x] `class MiniCPMRobotTrackModel(nn.Module, SupportsMultiModal)`,`is_pooling_model = True`,`@default_pooling_type(seq_pooling_type="LAST")`。
- [x] `__init__`:在 `_mark_language_model` 上下文里 `self.model = MiniCPMModel(vllm_config=vllm_config.with_hf_config(backbone_cfg), ...)`。
- [x] 移植 `VisionProjector` / `TemporalMarkerEncoder` / `control_query` / `FunnelTrajectoryHead`(结构与 checkpoint 逐名对齐,权重 1:1 加载)。
- [x] `output_scale` 按 `xy_scale` 重算并 `register_buffer(persistent=False)`。
- [x] `forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None)` → `self.model(...)`(`inputs_embeds` 分支绕开 `scale_emb`)。

### Phase 2 — Pooler(输出 24 维,**不归一化**)
- [x] **采用推荐 B**:forward 返回裸 hidden;`DispatchPooler({"embed": SequencePooler(LastPool(), head=self._pool_trajectory)})`,head 跑 `trajectory_head`(含 tanh×`output_scale`)→ `[num_seqs,24]`;`get_supported_tasks → {"embed"}`。head 用 bound method(不注册,避免与 `self.trajectory_head` 重复)。
- [ ] ~~方案 A(reward 模板)~~ **未采用**。仍确认避免 `pooler_for_embed`(会硬塞 `PoolerNormalize` 做 L2 归一化)。

### Phase 3 — 多模态输入管线(方案 A 核心)
- [x] `@MULTIMODAL_REGISTRY.register_processor(Processor, info=ProcessingInfo, dummy_inputs=DummyBuilder)`。
- [x] **modality key 复用 `"image"` + dict 值**(一次传 coarse/fine tokens + 两个 time_indices),自定义 `MultiModalDataParser._parse_image_data` 返回 `DictEmbeddingItems`。
- [x] `ProcessingInfo.get_num_image_tokens`:占位符数 = `(len(coarse)+#coarse段) + (len(fine)+#fine段) + 1(control)`(按张量 + time_indices 算)。
- [x] `_get_mm_fields_config`。→ **实际**:coarse/fine tokens 与 time_indices 用 `MultiModalFieldConfig.flat_from_sizes(...)`(不是 `batched`),另加 `coarse_lengths`/`fine_lengths` 走 `batched(...)` 供 embed 端切分。
- [x] `_get_prompt_updates`。→ **实际**:用 `PromptInsertion(target=PromptIndexTargets.end(), insertion=callable→[placeholder]*num)`(不是 `PromptReplacement`——本模型指令里没有占位 token,视觉块要追加到末尾)。
- [x] `DummyInputsBuilder`:造 dummy 特征张量 + 空文本(引擎 profiling 已实跑通过)。
- [x] `embed_multimodal`(旧名 `get_multimodal_embeddings`):跑 `VisionProjector` + 插 `temporal_markers` + 末尾填 `control_query`;按 `*_lengths` 逐项切分,返回与占位符逐位对齐的 embeds。
- [x] merge 走 `SupportsMultiModal.embed_input_ids` → `_merge_multimodal_embeddings`;**override `_embed_text_input_ids` 用 `self.model.embed_tokens`(不乘 scale_emb)**。
- [x] **占位符 token id**。→ **实际**:用**词表内保留 id `vocab_size-1`**(OOV id 会被输入校验拒绝),故**无需 `configure_mm_token_handling`**。
- [x] **prompt 顺序 = `[instruction_tokens, 占位块(末尾含 control)]`**,与 HF `[text, history, current, control]` 一致。

### Phase 4 — 权重加载
- [x] `load_weights` 用 `AutoWeightsLoader(self)` + `WeightsMapper(orig_to_new_prefix={"backbone.": "model."})`。
- [x] 主干沿用 qkv/gate_up 融合(`packed_modules_mapping` + `MiniCPMModel.load_weights` 的 stacked mapping)。
- [x] projector/marker/control_query/trajectory_head 直接加载;**`output_scale` 重算,加载时过滤掉 checkpoint 里的 `output_scale`**。

### Phase 5 — 配置 / 启动
- [x] `enable_mm_embeds=True`;`--limit-mm-per-prompt {"image":1}`(projector 在模型内跑,无外部 encoder)。已实跑通过。
- [x] `max_model_len` ≥ 拼装长度(实测真实尺寸 seq≈230;测试用 512)。

---

## 4. 正确性陷阱(务必逐条核对)

- [x] **⚠️ 文本 scale_emb 必须关**:override `_embed_text_input_ids`,传 `self.model.embed_tokens`(裸 embedding,不乘 `scale_emb=12`)。**已对齐,端到端 parity 通过。**
- [x] **control_query 是序列最后一位**(视觉块经 `PromptInsertion` 追加到末尾,LAST 池化取到它)。
- [x] **不做 L2 归一化**(head `activation` 未启用;未走 `pooler_for_embed`)。
- [x] **tanh + output_scale** 在头里逐元素做;`normalize_trajectory` 未接入(仅训练用)。
- [x] **marker 段计数**:`_count_marker_runs` 对相等 time_index 每个 run 插一个 marker;占位符长度精确复现(单测覆盖:coarse 155 + fine 65 + 1 = 221)。
- [x] Dropout 推理为 no-op(`model.eval()`),原样保留。

---

## 5. 待确认项(需模型侧 / 外部)

- [x] **projector/marker/head dtype**:对拍走**全 fp32** 通过;子模块在 `model_config.dtype` 下构造,边界显式 cast(投影按 projector dtype、head 按 head dtype)。生产可跑 bf16(未做 bf16 精度评测)。
- [ ] **真实视觉特征来源**:DINOv3+SigLIP 预处理管线仍不在本仓库;当前对拍用**合成特征**(两侧同源),生产精度评测需项目侧提供真实特征。**(仍开放)**
- [x] **占位符 token id**:定为 `vocab_size-1`(词表内保留特殊 token,自然语言指令不会命中)。

---

## 6. 测试计划(HF 参考 = golden,分层对拍)

- [x] **① 子模块单测**:VisionProjector / FunnelTrajectoryHead 加载同权重、形状 + 数值校验(见 `test_minicpm_robottrack.py`)。
- [ ] **② 主干 parity(独立脚本)**:未单独做——vLLM `MiniCPMModel` 需引擎上下文(attention backend/KV),难以脱离引擎单跑;**已由 ③ 端到端 parity 间接覆盖**(验证了 longrope/scale_depth/GQA + scale_emb 关闭)。
- [x] **③ 端到端 parity**:相同特征 → vLLM 24 维 reshape `[8,3]` vs HF `outputs.trajectories`,fp32 + `enforce_eager`,`allclose(rtol=2e-3, atol=2e-3)` **通过**,`max abs diff≈5.9e-4`;另验 batch=2 逐项切分正确、无串扰。
- [x] **④ 离线入口 / 解耦**:用子模块单测解耦「移植正确性」;MM 管线用引擎 e2e 实跑验证(processor 占位符数、字段切分均通过)。golden 生成脚本 `/tmp/gen_golden.py`(4.56.2 环境)、vLLM 端 `/tmp/test_vllm_e2e.py`(主 venv)。
- [ ] golden 存取模板(`test_reward.py` 的 `dump_reward_outputs`):**未采用**,用了独立复现脚本(待整理入库)。
- [x] MM+pooling 测试参考 `test_clip.py`(`vllm_runner(runner="pooling")`)——门控 e2e 测试已按此写。
- [x] 加 `tests/models/registry.py` 条目 + pooling parity 测试(纯逻辑单测 + 门控 e2e）。
- [ ] PR 说明:非重复性、测试命令与结果、模型评测、声明 AI 辅助(见 `AGENTS.md`)。**(尚未开 PR)**

---

## 7. 模板参考地图(file:line)

| 用途 | 文件 | 行 |
|---|---|---|
| 主干 `MiniCPMModel` / forward / scale_emb | `models/minicpm.py` | 401 / 445-456 / 441-443 |
| qkv·gate_up 融合 + load_weights | `models/minicpm.py` | 487-494 / 559-569 / 650-655 |
| `with_hf_config` | `config/vllm.py` | 717 |
| 子配置用法 | `models/idefics3.py`, `granite4_vision.py` | 478 / 536 |
| config dict→对象包装 | `transformers_utils/configs/kimi_vl.py`, `granite4_vision.py` | — / 78-92 |
| `_CONFIG_REGISTRY` | `transformers_utils/config.py` | 72-132 |
| reward 头/分工模板 | `models/qwen2_rm.py` | 50-66 / 74-86 / 93-103 |
| Pooler ABC / LastPool / SequencePooler | `layers/pooler/abstract.py`, `seqwise/methods.py`, `seqwise/poolers.py` | 16-36 / 50-57 / 88-98 |
| EmbeddingPoolerHead / IdentityPooler | `seqwise/heads.py`, `pooler/special.py` | 33 / 140 |
| `PoolingOutput.data` | `outputs.py` | 66-82 |
| MM: image_embeds 输入/解析/处理 | `models/qwen2_vl.py` | 147-173 / 1300-1322 / 1348-1369 |
| MM: processor/info/dummy/fields/prompt | `models/glm4_1v.py` | 1760 / 978-1163 / 1460-1547 / 1684 / 1693-1757 |
| MM: dict 特征解析 override | `models/minicpmv.py` | 514-537 |
| MM: merge / embed_input_ids | `models/utils.py`, `models/interfaces.py` | 637-673 / 380-415 |
| MM: modality key / enable_mm_embeds | `inputs/llm.py`, `config/multimodal.py` | 29-51 / 97 |
| MM+pooling 共存范例 | `models/clip.py` | 768-975 |
| HF 源模型 | `~/model/MiniCPM-RobotTrack/modeling_robottrack.py`, `config.json` | — |
