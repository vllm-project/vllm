# MiniCPM-RobotTrack → vLLM 集成实现报告

> 配套文档:`MiniCPM-RobotTrack-vLLM-TODO.md`(分阶段清单 + 模板参考)。
> 本文汇总**已实现内容、设计决策、验证结果、运行方式、已知限制**。
> 状态:**已完成并端到端验证**(含合成特征对拍 + 真实图片跑通)。

---

## 1. 目标与结论

把 `MiniCPM-RobotTrack`(基于 MiniCPM4-0.5B 的 VLA 轨迹策略模型)作为 **in-tree pooling 模型**接入本 vLLM 分支。模型是**非生成式**:单次 causal 前向 → 取序列最后一位(control_query)→ 轨迹头回归 **8 个 `[x, y, yaw]` 路点**,因此映射到 vLLM 的 **pooling / `embed`** 范式,而非 `generate`。

**结论**:实现完成,端到端输出与 HF 参考 `torch.allclose(rtol=2e-3, atol=2e-3)` 一致(`max abs diff ≈ 5.9e-4`);真实图片(DINOv3+SigLIP)→ vLLM 逐帧推理 → 可视化 demo 跑通。不依赖 `trust_remote_code`。

---

## 2. 落地文件

### 新增
| 文件 | 作用 |
|---|---|
| `vllm/transformers_utils/configs/minicpm_robottrack.py` | `MiniCPMRobotTrackConfig`:把 `backbone_config` dict 包成 `PretrainedConfig`,注入 `rope_theta`,`get_text_config()` 返回 backbone |
| `vllm/model_executor/models/minicpm_robottrack.py` | 模型本体 + 自定义轨迹 Pooler + MM processor/info/dummy/parser |
| `tests/models/multimodal/pooling/test_minicpm_robottrack.py` | 纯逻辑单测 + 门控 e2e 测试 |
| `examples/pooling/robottrack_minicpm_offline.py` | 离线 demo(合成特征,展示 pooling 接口) |
| `examples/pooling/robottrack_minicpm_video.py` | 真·图片输入 demo(DINOv3+SigLIP → 32 帧 cache → 逐帧推理 → BEV 叠加) |

### 修改(仅注册,共约 11 行)
| 文件 | 改动 |
|---|---|
| `vllm/transformers_utils/config.py` | `_CONFIG_REGISTRY` 加 `minicpm_robottrack` |
| `vllm/transformers_utils/configs/__init__.py` | 导出 `MiniCPMRobotTrackConfig` |
| `vllm/model_executor/models/registry.py` | `_EMBEDDING_MODELS`(多模态段)加 `MiniCPMRobotTrackModel` |
| `tests/models/registry.py` | 加测试注册条目 |

---

## 3. 架构映射(HF → vLLM)

| HF 模块 | vLLM 落点 |
|---|---|
| `backbone`(MiniCPMModel) | 复用 `minicpm.py: MiniCPMModel`,经 `vllm_config.with_hf_config(backbone_config)` |
| `vision_projector` | 移植为 `nn.Module`(结构与 checkpoint 逐名对齐) |
| `temporal_markers` | 移植 `nn.Embedding×3`,在 `embed_multimodal` 内按帧插 marker |
| `control_query` | `nn.Parameter`,填到视觉块末尾(保证是序列 LAST) |
| `trajectory_head` | 移植(`nn.Linear`+GELU+tanh) |
| `output_scale` | 按 `xy_scale` **重算**,加载时过滤 checkpoint 里的该 buffer |
| 读 last hidden | 自定义 `Pooler`:`LastPool` → 轨迹头 → `[n,24]` |

### 关键设计决策(与踩坑)
1. **config 包装**:`minicpm` 不在 `transformers.CONFIG_MAPPING`,用裸 `PretrainedConfig(**bc)`(transformers 5.x 会自动 `rope_scaling→rope_parameters`);**必须在构造后注入 `rope_theta=10000.0` 到 `rope_parameters`**,否则 vLLM LongRoPE 的 base 为 `None` 会崩。
2. **文本 scale_emb 必须关**(首要数值对齐项):override `_embed_text_input_ids`,传 `self.model.embed_tokens`(裸 embedding),绕开 `MiniCPMModel.embed_input_ids` 的 `×scale_emb=12`。HF RobotTrack 文本走 `inputs_embeds`,不乘 scale。
3. **Pooler 不做 L2 归一化**:`DispatchPooler({"embed": SequencePooler(LastPool(), head=self._pool_trajectory)})`,head 跑 `trajectory_head`(含 `tanh × output_scale`)→ flatten `[n,24]`;head 用 bound method,避免与 `self.trajectory_head` 重复注册。**不用 `pooler_for_embed`**(它会塞 `PoolerNormalize`)。
4. **占位符 token**:用**词表内保留 id `vocab_size-1`**;OOV id(如 128000)会被输入校验拒绝(`> max(tokenizer.max_token_id, model_vocab_size-1)`)。
5. **权重加载**:`AutoWeightsLoader(self)` + `WeightsMapper(orig_to_new_prefix={"backbone.": "model."})`;主干 qkv/gate_up 融合由 `MiniCPMModel.load_weights` 处理。

---

## 4. 多模态输入管线(方案 A:预计算特征 in)

模型吃**预计算的 1536 维 DINOv3+SigLIP 融合特征**,不吃像素;视觉编码在 vLLM 之外。

- **数据形态**:`"image"` 模态传 dict —— `coarse_tokens[N_c,1536]`、`coarse_time_indices[N_c]`、`fine_tokens[N_f,1536]`、`fine_time_indices[N_f]`。
- **Parser**:自定义 `MultiModalDataParser._parse_image_data` → `DictEmbeddingItems`。
- **字段**:coarse/fine 的 tokens 与 time_indices 用 `flat_from_sizes`,另加 `coarse_lengths`/`fine_lengths` 走 `batched`,`embed_multimodal` 按其切分(对「拼接张量 / 逐项列表」两种投递都鲁棒)。
- **占位符注入**:本模型无 HF image processor,`_call_hf_processor` 只做文本分词;视觉块用 `PromptInsertion(target=PromptIndexTargets.end())` 追加到指令后 —— 与 HF `[text, history, current, control]` 顺序一致。
- **占位符数量** = `(len(coarse)+#coarse段) + (len(fine)+#fine段) + 1(control)`;真实尺寸下 = `124+31 + 64+1 + 1 = 221`(已单测校验)。
- **合并**:`embed_multimodal` 跑 projector + 插 temporal marker(history `stream_id=0` / current `stream_id=1`)+ 末尾填 control_query;经 `SupportsMultiModal.embed_input_ids` → `_merge_multimodal_embeddings` 散射到占位符位。

---

## 5. 验证

| 层级 | 方法 | 结果 |
|---|---|---|
| ① 子模块 | VisionProjector / FunnelTrajectoryHead 构造 + 加载 HF 权重 + 形状/数值 | ✅ |
| ② 主干 parity(独立) | 未单独做(vLLM 主干需引擎上下文),由 ③ 间接覆盖 | — |
| ③ 端到端 parity | 同一合成特征,vLLM 24 维 vs HF golden | ✅ `allclose(2e-3)`,`max diff 5.9e-4` |
| batch | 2 条请求逐项切分、无串扰,item0 仍等于 golden | ✅ |
| MM processor | 占位符数量 / 字段 batching | ✅ 221 |
| 真实图片 | `track-image/0`(139 帧)逐帧跑通 + 轨迹对场景有响应 | ✅ |
| Lint | `ruff check` / `ruff format` / `mypy` | ✅ |

**golden 生成注意**:HF 远程代码只兼容 `transformers>=4.56,<5`,本仓库 transformers 5.14 会让远程前向出 NaN / 导入失败。golden 在**隔离的 `transformers==4.56.2` 环境**生成(`uv pip install --python <venv> transformers==4.56.2`,复用主 `.venv` 的 torch),vLLM 结果在主 `.venv`(5.14)生成,同输入、fp32 对拍。详见 TODO 文档 `✅.1` 小节。

---

## 6. 运行方式

### 6.1 离线(合成特征,验证 pooling 通路)
```bash
python examples/pooling/robottrack_minicpm_offline.py \
    --model /path/to/MiniCPM-RobotTrack --dtype float32 --gpu-memory-utilization 0.35
```

### 6.2 真·图片输入 + BEV 可视化
```bash
CUDA_VISIBLE_DEVICES=6 python examples/pooling/robottrack_minicpm_video.py \
    --model  /cache/zhanghao/model/MiniCPM-RobotTrack/ \
    --dino   /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
    --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
    --images track-image/0 --output output --bev-range 1.0 \
    --instruction "Follow the person."
```
链路:每帧 resize 384 → DINOv3(384)+SigLIP(1152) 拼接=1536 → 网格 24×24 → pool 成 coarse(4)/fine(64) → `deque(maxlen=32)`(31 coarse history + 1 fine current,不足补齐)→ `llm.embed` → 8 路点 → BEV 叠加(x=前进朝上、+y=左)→ 存 `output/frame_XXXXX.jpg`。

### 6.3 参考:vLLM 引擎入口
```python
llm = LLM(model=..., runner="pooling", enable_mm_embeds=True,
          limit_mm_per_prompt={"image": 1}, dtype="float32", enforce_eager=True)
out = llm.embed([{"prompt": instr, "multi_modal_data": {"image": {...}}}])
traj = torch.tensor(out[0].outputs.embedding).reshape(8, 3)
```

---

## 7. DINOv3 + SigLIP 预处理链路(在 vLLM 之外)

来自上游工程 `~/MiniCPM-Robot/MiniCPM-RobotTrack/minicpm_robot_track/`:

1. 每帧 **resize 到 384×384**(Go2 实机是 center-crop 384)。
2. **DINOv3**(`facebook/dinov3-vits16-pretrain-lvd1689m`,384 维,ViT-S/16):取 `last_hidden_state`,丢 CLS + 4 register token → 24×24=576 patch。
3. **SigLIP**(`google/siglip-so400m-patch14-384`,1152 维):取 `last_hidden_state`,adaptive-pool 到 24×24。
4. **通道拼接:`cat((dino, siglip))` = 1536,DINO 在前**。
5. 网格 24×24 → adaptive_avg_pool 成 **coarse=2×2=4** 和 **fine=8×8=64** token(每帧两份)。
6. 时序拼装(`data.py`):current 用 fine(time=31),history 取最近 31 帧的 coarse(time=0..30),不足则复制最旧帧补齐到 31。

**目的**:SigLIP 提供语义/语言对齐(按指令锁目标),DINOv3 提供空间/几何(精确定位跟踪),互补支撑「语言条件目标跟踪」。

**vLLM 侧不含也不需要这两个 encoder**;`vllm` 原生有 SigLIP,但**没有 DINOv3**(仅 OpenVLA 里用 `timm` 的 DINOv2)。若要「图片直接进 vLLM」(方案 B),需以 OpenVLA 为骨架另接 DINOv3+SigLIP,属独立更大工作项。

---

## 8. 已知限制 / 待办

- [ ] **真实视觉特征的训练一致性**:video demo 的内联编码器忠实复刻了上游逻辑,但**未做逐值 parity**(register-token 丢弃、拼接顺序、池化都对齐了)。严格对齐建议加「内联 vs 上游 `DualVisionEncoder`」的 allclose 校验。
- [ ] **② 主干独立对拍脚本**未单独做(由 ③ 覆盖)。
- [ ] **BEV 坐标约定**按 REP-103(x 前进、+y 左);若部署端 y 正方向相反,改 `to_px` 一个符号。
- [ ] **PR**:尚未开;按 `AGENTS.md`,AI 辅助工作需人工逐行 review、跑测试,PR 描述含非重复性/测试结果/模型评测/AI 辅助声明。
- [ ] **KV cache**:策略模型无状态单前向、`use_cache=False`,不跨帧复用 KV;实机跨帧复用发生在「视觉特征」层(`VisionFeatureCacher`),非 KV。
- [ ] 复现脚本目前在 `/tmp`(`gen_golden.py` / `test_vllm_e2e.py`),可按需整理入库。

---

## 9. 版本 / 环境备注

- 主环境:本仓库 `.venv`,transformers **5.14.1**,torch **2.11.0+cu130**。
- golden 环境:隔离 `transformers==4.56.2`(仅用于生成 HF 参考)。
- DINOv3/SigLIP 在 transformers 5.14 下均可加载(`dinov3_vit` / `siglip` 均在 `CONFIG_MAPPING`)。
