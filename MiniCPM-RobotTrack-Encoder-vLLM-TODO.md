# DINOv3+SigLIP Encoder → vLLM 集成 TODO

> 目标:把 MiniCPM-RobotTrack 的视觉编码器(**DINOv3 + SigLIP 融合**)从「vLLM 之外的预处理」搬进 vLLM,做成**模型内的 vision tower**(方案 B / 「图片 in」),让 vLLM 直接吃图片、吐 8 路点。
> 现状是**方案 A(特征 in)**,已完成并验证(见 `MiniCPM-RobotTrack-vLLM-IMPLEMENTATION.md`)。本文只规划**方案 B**。
> 主模板:`vllm/model_executor/models/openvla.py`(已有的 fused DINOv2+SigLIP VLA 骨干)。

---

## ✅ 初步版本(已实现,2026-07-23)

**目标达成**:DINOv3+SigLIP 编码器已搬进 vLLM(模型内 vision tower);**32 帧滚动窗口仍由客户端维护**;客户端每步只发**原始帧**,vLLM 内部完成 resize/归一化 + 编码 + 池化 + 时序拼装。

- **走的是 B-1a(整窗口 = 一个 mm item),不是 B-2**;这是 TODO 认定的「最快跑通」退路,契合「初步版本 + 客户端维护 deque」。
- **tower 完全 vLLM 原生(2026-07-23 升级)**:两个编码器都是 in-tree vLLM 模块,不再用 transformers 编码器模型——SigLIP 复用 vLLM 原生 `SiglipVisionModel`,DINOv3 新写 `vllm/model_executor/models/dinov3.py`(vLLM TP linear + SDPA + 忠实 2D RoPE,对 HF RoPE 逐值 0.0)。详见 `MiniCPM-RobotTrack-Encoder-vLLM-IMPLEMENTATION.md` §5.2。
- **客户端契约 = 传原始帧**(P1):`mm_data={"image": {"frames": [<=32 帧原始 HxWx3>]}}`;processor 内跑 DINO/SigLIP image processor(resize384+各自归一化),tower 跑两个编码器。相较「客户端归一化后发像素」省 ~16× IPC。
- **无 tower 内缓存**:每步重编码整窗口(DINOv3-S+SigLIP 不大,先测瓶颈;要省重编码应升级到 B-2 命中框架 `encoder_cache`)。
- **双模、向后兼容**:features-in(方案 A)路径完整保留;parser 按 dict 内容分派(`coarse_tokens` → 特征;`frames` → 像素)。

**落地文件**:
- `vllm/transformers_utils/configs/minicpm_robottrack.py`:新增 `dino_model` / `siglip_model` / `image_size`(经 `hf_overrides` 注入本地路径)。
- `vllm/model_executor/models/minicpm_robottrack.py`:`DualVisionTower`(`_mark_tower_model` 内构造、`from_pretrained` 自加载、`load_weights` 补报已加载)、`MiniCPMRobotTrackPixelItems`、parser 分派、`_call_hf_processor` 跑图像处理器、fixed-221 占位符、`_encode_window`/`_embed_pixel_windows`(复用 `_embed_visual_bundle`)、pixels-in dummy(profiling 覆盖 tower)。
- `examples/pooling/robottrack_minicpm_video.py`:客户端只维护 `deque(maxlen=32)` 原始帧、传窗口;`LLM(..., hf_overrides={dino_model, siglip_model, image_size})`。
- `tests/models/multimodal/pooling/test_minicpm_robottrack.py`:新增 `_square_side`/`_grid_pool`/`_pad_history_frames`/窗口 221 不变量单测。

**运行**:
```bash
CUDA_VISIBLE_DEVICES=6 python examples/pooling/robottrack_minicpm_video.py \
    --model /cache/zhanghao/model/MiniCPM-RobotTrack/ \
    --dino  /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
    --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
    --images track-image/0 --output output --bev-range 1.0
```

**已知限制 / 下一步**:
- [x] **编码器 parity 已对拍**(2026-07-23):同一批帧,pixels-in(内嵌 tower)vs features-in(上游忠实 `DualVisionEncoder`),vLLM 24 维输出 **max abs diff 3.0e-3**(8 帧;frame0 完全一致 0.0,其余为两次独立 fp32 DINO/SigLIP 前向的数值噪声);features-in 此前已对 HF golden 验证,故等价链成立。逻辑逐项复刻(丢 CLS+register、DINO 在前拼接、SigLIP 网格 adaptive-pool 到 24×24、coarse4/fine64、resize384 BICUBIC)。
- [ ] **同批次混用 pixels-in 与 features-in 不支持**(按是否含 `dino_pixels` 整批分派);单客户端单模,无影响。
- [ ] **无 encoder cache**:每步重编码整窗口;要复用历史帧编码 → 升级 B-2(每帧一 item + marker-item,命中框架 `encoder_cache`)。
- [ ] tower 默认 eager/fp32,未接 encoder CUDA graph;性能对比方案 A 待做(Phase 7)。

---


## 0. 是否要做(先决策)

- ✅ **服务端 / 大 GPU / 多路并发**:值得做,吃到 vLLM 的 encoder 缓存 + GPU 批处理 + 单一部署产物。
- ❌ **边缘 / 机器人(Go2)/ 追求与训练一致**:建议**别做**,保持方案 A,编码器放独立 TensorRT/ONNX 服务(上游已有 `build_engines.sh`)。这是上游实机的选择。
- ⚠️ 复杂度真实:要封 timm DINOv3 + SigLIP tower、写 image processor、并在 `embed_multimodal` 里重建结构化时序序列。**没决定要上生产前,建议先做原型评估。**

---

## 1. 已锁定的架构决策

- [x] encoder = **vision tower(模型子模块,ModelRunner GPU 执行)**,**不是 preprocessor**(preprocessor 是 CPU 输入准备,吃不到 ModelRunner 特性)。
- [x] **tower 必须无状态、逐帧纯函数**(pixels→features);否则 encoder cache(按 mm_hash)、批处理、并发多路全崩。
- [x] **32 帧滚动窗口 = 客户端/会话层维护**(deque),不进 tower、不进引擎;每次请求携带自己的窗口 + 描述符(谁 current、time_indices)。
- [x] **跨帧复用靠「内容寻址」**(同帧哈希→同特征),不靠 tower 里的流式状态。
- [x] DINOv3 走 **timm**(vLLM 无原生 DINOv3;OpenVLA 的 DINOv2 也是 timm);SigLIP 用 **vLLM 原生 `siglip.py`** 或 timm。
- [x] 结构化组装(coarse/fine 角色 + marker + control_query + time_indices)在 `embed_multimodal` 里做——**逐请求纯计算,不跨请求留状态**。
- [x] **双模、向后兼容**:B-2 只**新增** pixels-in 路径,**保留现有 features-in(方案 A)路径**(照 `qwen2_vl` 的 `pixel_values` OR `image_embeds` 惯例)。
  - > **备注(兼容性)**:实现 B-2 **不破坏现有「客户端自己 encode、发特征」的客户端**——它继续走 features-in 即可,不用改。features-in 路径**维持「整窗口一个 dict」**、**不受 B-2 逐帧 item 拆分影响**(该路径服务端无编码,用不到 encoder cache;重用由客户端上游 `VisionFeatureCacher` 负责)。B-2 的逐帧 item + marker-item + encoder cache **只服务 pixels-in**。唯一约束:客户端若从 features-in 切到 pixels-in,vLLM tower 的编码须与原 encode **逐项一致**(见 §6 parity)。

---

## 2. 关键设计难点(RobotTrack ≠ 单图 VLM)

vLLM 标准 mm 模型是「一个 item → 一段固定占位符,scatter 一份 embedding」。但 RobotTrack 是**顺序敏感的结构化序列** `[history粗+marker, current细+marker, control]`,同一帧作 history 用 coarse(4)、作 current 用 fine(64),角色/time_index 随窗口滚动而变。两种映射方案:

### 方案 B-1(更快跑通,但缓存非惯用):**整窗口 = 一个 mm item**
- `mm_data["image"]` = 32 帧像素 + 描述符(哪个是 current、time_indices)。
- 一次请求一个 item → `embed_multimodal` 收到全部帧,内部批量过 tower,拼出完整 `[~221, hidden]` 序列。
- 组装自然(和现在方案 A 一样,只是把编码放进来)。
- 缺点:vLLM 的**逐 item encoder cache 帮不上**(整窗口一个哈希,每步都变 → 每步重编码 32 帧)。→ 想省重编码只能在 **tower 内部自建「帧特征 LRU,按帧内容哈希」**。
- > **⚠️ 备注(重要)**:**tower 内自建 GPU 特征缓存不符合 vLLM 一贯设计,仓库里也无先例**。vLLM 的惯例是「**框架**按内容寻址、统一管生命周期」来缓存(`encoder_cache`/`mm_processor_cache`/KV/Mamba state),**模型本体保持无状态**;模型里的 `lru_cache` 全是确定性小工具(rope/位置 id/预处理常量),没有「按图片内容缓存编码结果」的。自建缓存会踩几条假设:CUDA graph 假设无副作用、TP/多副本各持一份缓存不一致、显存不计入框架 encoder budget、绕过框架哈希/校验。→ **属「偏离设计的务实 hack」,若走 B-1 要么接受非惯用并限定单 worker,要么干脆不缓存、每步重编码(DINOv3-S+SigLIP 不大,先测瓶颈再定)。**

### 方案 B-2(vLLM 惯用,建议直接做):**一帧 = 一个 mm item**
- 每帧独立 item、哈希按(像素+角色)→ 命中 vLLM 原生 `encoder_cache`(每步仍带整窗口,只是历史帧 item 命中缓存跳过重编码)。**这才是 vLLM「框架管缓存、模型无状态」的设计意图。**
- 真正的难点**不是「变长」**(vLLM 本就支持不同 item 不同 token 数),而是 **marker/control 是「item 之间的粘合物」、embedding 是算出来的(非词表),且 marker 随窗口滚动**。切法决定能否命中缓存:
  - ❌ 把 marker 折进每帧 item(`[m_i, c_i×4]`)→ 吻合 scatter,但 **marker 依赖 time_index、每步变 → item 哈希变 → 原生 cache 全 miss**,省重编码的初衷落空;
  - ✅ **推荐**:**帧-item(像素+角色,可缓存)与 marker-item(位置,固定集)/control-item 拆开、交错排** → 贵的 DINO+SigLIP 命中缓存复用,便宜的 marker 每步重算,排布每步由 processor 重建(见 §3.B-2);
  - 或在 gather 之后插 marker/control → 需改 vLLM merge 逻辑(不推荐)。
- 成本:一请求要摆较多碎 item(帧+marker+control)、占位符 bookkeeping 繁琐;但**不需要模型内状态、缓存全交框架**,是长期正确的方向。

> **决策(更新)**:**优先直接实现 B-2**(vLLM 惯用:框架 `encoder_cache` 管缓存、tower 无状态),按「帧-item / marker-item / control-item 拆分」的切法。
> **B-1 仅作退路**:想最快跑通、且暂不在意重编码开销时用它(不加缓存);**B-1 + tower 内缓存不推荐**(非惯用,见上备注)。

---

## 3. 数据流

### 3.A 数据流(B-1,退路)

> **客户端默认传「像素」**,不是 frame_id。vLLM 的 mm 管线要求真实张量数据流经模型;
> 模型只能看到 `mm_kwargs` 里的东西,拿 opaque frame_id 无法在缓存 miss 时恢复像素。
> frame_id 仅在「自建帧像素库 + tower 按 id 查缓存」时才有意义,但会破坏内容寻址、
> 让 tower 变有状态、在 eviction/多副本下出错 —— **默认不做(见下 B-1a vs B-1b)**。

```
客户端/会话层:
  deque(maxlen=32) 存帧(PIL/像素)
  每帧 -> 组 request: {prompt,
                       mm_data={"image": {pixels[<=32], current_idx, time_indices}}}

vLLM(模型内):
  processor: 像素准备(resize384 + DINO/SigLIP 各自归一化)
  embed_multimodal:
    for 每帧: fused = concat(DINO(384), SigLIP(1152))
              [tower 内「按帧内容哈希」LRU:命中则跳过 DINO+SigLIP 重编码]
    history 帧 -> pool 2x2(coarse4); current 帧 -> pool 8x8(fine64)
    插 temporal marker(history stream0 / current stream1,time_index 来自描述符)
    末尾拼 control_query
    -> [~221, hidden] 对齐占位符
  backbone + trajectory head(已实现,不动)
```

**B-1a(推荐)**:每步传整窗口 32 帧像素;tower 按**帧内容哈希**缓存,省**重编码**(不省重传)。内容寻址、无状态、并发/副本安全。
**B-1b(不推荐)**:每步只传 current 像素 + history 的 frame_id;省重编码**和**重传,但 tower 按 id 有状态、破坏内容寻址、eviction/多副本会崩。
**若真要「省重编码又不想自建缓存」→ 应走 B-2**(每帧一个 item,命中 vLLM 原生 `encoder_cache`),代价是组装更难(见 §2)。

### 3.B 数据流(B-2,首选实现)

> 关键:把「随窗口滚动的关联」和「稳定可缓存的内容」拆到不同 item。
> **帧-item = (像素 + coarse/fine 角色)**,不含 time_index → 哈希稳定 → 框架 `encoder_cache` 命中(一帧在 history 期间约 31 步复用同一份 coarse)。
> **marker-item = (位置)**,值是固定集(0..30 + current),常量/廉价;**control-item = 常量**。
> 每步变的只是「哪帧摆哪个位置槽」= **占位符排布**,由 processor 每请求重建(CPU,便宜)。

```
客户端/会话层:
  deque(maxlen=32) 存帧(PIL/像素)
  每帧 -> 组 request: {prompt, mm_data={"image": [<=32 帧,各带 role(coarse/fine)]},
                       并给出每帧的位置(time_index)用于排布 marker}

vLLM(模型内):
  processor: 把窗口铺成交错的 per-item 占位符:
             [marker_pos0][frame0][marker_pos1][frame1]...[marker_cur][frame_cur][ctrl]
             (每个 item 各一段占位符;帧-item 的 mm_hash 只按 像素+role,不含位置)
  embed_multimodal(逐 item):
    frame-item -> DINO+SigLIP -> pool(coarse/fine)   [框架 encoder_cache 按 mm_hash 命中复用]
    marker-item -> time_emb(pos)+stream+camera        (常量集,不缓存也廉价)
    control-item -> control_query                     (常量)
  merge: 各 item embedding 各填自己那段占位符(vLLM 标准 scatter)
  backbone + trajectory head(已实现,不动)
```

- **好处**:tower 无状态、缓存全交框架(惯用);贵的编码复用、便宜的 marker 重算。
- **成本**:一请求 item 数多(帧+marker+control);processor 的交错占位符 bookkeeping 要写对(**首要正确性风险**)。
- **缓存 caveat**:`encoder_cache` 按调度生命周期释放,非常驻;命中率取决于容量(≥历史帧数)与相邻请求的时序。


---

## 4. 分阶段实现清单

> **目标 = B-2**(帧-item / marker-item / control-item 拆分,缓存交框架 `encoder_cache`,tower 无状态)。
> 下方凡标「(B-1 退路)」的项仅在改走 B-1 时才做;B-2 不做 tower 内缓存。

### Phase 0 — config
- [ ] 给 `MiniCPMRobotTrackConfig` 增补 vision 段:`dino_model`、`siglip_model`、`image_size=384`、`coarse_tokens_per_frame`、`fine_tokens_current_frame`(部分已有);或新增 `VisionEncoderConfig`。
- [ ] 决定 encoder 权重来源(外部 HF/timm 路径),写清依赖(`timm`、`transformers` 的 dinov3/siglip)。

### Phase 1 — vision tower(nn.Module)
- [ ] `class DualVisionTower(nn.Module)`:DINOv3(timm)+ SigLIP(vLLM 原生或 timm),`_mark_tower_model(vllm_config, "image")` 内构造(模板 `openvla.py:84/405`)。
- [ ] `forward(pixels)`:丢 CLS+register → 24×24 网格;SigLIP 网格 adaptive-pool 到 24×24;**`cat((dino, siglip))=1536`,DINO 在前**;返回融合网格(role-independent)。
- [ ] `pool_coarse/pool_fine`:2×2 / 8×8 adaptive_avg_pool。
- [ ] **frame-item 无状态**:tower 只是 `pixels(+role) -> pooled` 的纯函数,缓存交框架 `encoder_cache`(按 mm_hash)。
- [ ] ~~帧特征 LRU~~ **(B-1 退路,不推荐)**:仅当走 B-1(整窗一个 item)时才在 tower 内自建内容哈希缓存;**非惯用,见 §2 备注**。

### Phase 2 — image processor(B-2:交错 per-item 占位符)
- [ ] 自定义 processor:PIL→像素,resize 384,DINO/SigLIP 各自归一化(OpenVLA 打包 6 通道 `openvla.py:163-171`;也可传两张 tensor)。
- [ ] **把窗口铺成交错 item**:`[marker_pos][frame(coarse/fine)]…[marker_cur][frame_cur][ctrl]`,每 item 一段占位符;**frame-item 的 mm_hash 只按 像素+role,不含 time_index/位置**(保证跨步命中缓存)。
- [ ] `_get_mm_fields_config`:各 item 字段;marker-item 的位置、control-item 作为独立 item。
- [ ] `_get_prompt_updates`:按上面交错顺序插各 item 的占位符(总数 = Σ(1 marker + coarse/fine) + 1 control)。
- [ ] DummyInputsBuilder:造 dummy 帧 + 交错布局,过引擎 profiling。

### Phase 3 — embed_multimodal(逐 item,B-2)
- [ ] **逐 item 返回**:frame-item → tower(框架 `encoder_cache` 命中)→ 按 role 做 coarse/fine pool;marker-item → `time_emb(pos)+stream+camera`;control-item → `control_query`。各返回自己那段 embedding,vLLM 标准 scatter 填占位符。
- [ ] marker 的 embedding 计算可**复用现有 `minicpm_robottrack.py: _insert_temporal_markers` / control 逻辑**(拆成逐 item 版)。
- [ ] 顺序/对齐由 processor 的交错占位符保证;backbone/pooler/head 不变(仍「特征之后」)。

### Phase 4 — 客户端/会话层
- [ ] deque(maxlen=32)维护帧 + 每请求组窗口 + 描述符(current、time_indices,按 `data.py:212-245` 逻辑,history 不足复制最旧帧补齐 31)。
- [ ] 每请求把窗口铺成交错 item;**传像素**(见 §3 备注,默认不用 frame_id)。

### Phase 5 — 缓存策略
- [ ] 明确「帧特征 LRU」的 key(帧内容哈希 / frame_id)与容量(≥32);验证并发多路互不污染。
- [ ] 评估是否值得走 B-2(vLLM 原生 encoder cache);记录 marker/control 落位的改动点。

### Phase 6 — 验证
- [ ] **编码器 parity**:内嵌 tower vs 上游 `DualVisionEncoder`,同图 `torch.allclose`(register 丢弃、拼接顺序、pool 网格对齐)。
- [ ] **端到端 parity**:图片 in → vLLM 24 维 vs 上游「vision→features→policy」全链路(fp32)。
- [ ] 复用现有 `track-image/0` 逐帧跑通 + BEV 可视化(demo 复用 `robottrack_minicpm_video.py`,把 encoder 从客户端移到模型内)。

### Phase 7 — 性能(可选)
- [ ] encoder dtype(fp16/bf16)、encoder CUDA graph(`encoder_cudagraph_manager`,静态 shape 才行)。
- [ ] 对比方案 A(外部 TensorRT 编码器)吞吐 / 显存 / 延迟,给 go/no-go 结论。

---

## 5. 正确性陷阱

- [ ] **tower 必须无状态**(纯函数 `pixels+role -> pooled`;缓存交框架,不在模型里按流位置存)。
- [ ] **marker 依赖 time_index/位置,随窗口滚动** → **B-2:marker 单独成 item**(常量集,不进 frame-item 的哈希),保证 frame-item 跨步命中缓存;**别把 marker 折进 frame-item**(否则哈希每步变、缓存全 miss)。
- [ ] **frame-item 的 mm_hash 只按 像素+role,不含 time_index/位置**(否则跨步不命中)。
- [ ] **拼接顺序 DINO 在前、SigLIP 在后**;网格 24×24;coarse=2×2、fine=8×8;resize 384(实机 center-crop 384)——必须与训练/上游逐项一致。
- [ ] **DINOv3 丢 CLS + register token**(vits16 有 4 个 register);SigLIP 若非完全平方 patch 数要丢首 token 再 pool。
- [ ] control_query 仍是序列最后一位;LAST 池化取它。
- [ ] history 不足 31 帧时复制最旧帧补齐(`data.py:232-234`)。

---

## 6. 好处 / 坏处(决策依据)

**好处**:单一部署产物;历史帧编码可复用(帧缓存 / 或 vLLM encoder cache);图片进路点出、无特征搬运;vLLM 统一显存 + GPU 批处理 +(部分)encoder CUDA graph。

**坏处**:复杂度/维护高(封 tower + processor + 结构化组装);偏离 checkpoint(需外部编码器权重);丢上游 TensorRT 优化路径(tower 默认 eager);流式状态仍在客户端;显存争用;版本矩阵更复杂(vLLM+transformers+timm+权重);编码器 vs 策略 bug 更难隔离。

---

## 7. 模板参考(file:line)

| 用途 | 文件 | 行 |
|---|---|---|
| vision tower / `_mark_tower_model` / `embed_multimodal` | `vllm/model_executor/models/openvla.py` | 84 / 405 / 451-464 |
| 6 通道 dino/siglip 拆分 + concat | `openvla.py` | 163-176 |
| processor / pixel_values 字段 | `openvla.py` | 315-328 |
| vLLM 原生 SigLIP tower | `vllm/model_executor/models/siglip.py` | — |
| encoder cache(按 mm_hash)/ encoder cudagraph | `vllm/v1/worker/gpu_model_runner.py` | 3165 / ~3147 |
| 上游编码器(DINOv3+SigLIP 融合) | `~/MiniCPM-Robot/MiniCPM-RobotTrack/minicpm_robot_track/vision.py` | — |
| 上游时序拼装(history/current/time) | `~/MiniCPM-Robot/.../minicpm_robot_track/data.py` | 212-245 |
| 现成的结构化组装(方案 A,可搬用) | `vllm/model_executor/models/minicpm_robottrack.py` | `embed_multimodal` / `_insert_temporal_markers` |
| 现成图片 demo(把 encoder 移进模型即可) | `examples/pooling/robottrack_minicpm_video.py` | — |
