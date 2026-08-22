# MiniCPM-RobotTrack 逐帧特征缓存 → vLLM 实现报告

> 配套文档:`MiniCPM-RobotTrack-Encoder-vLLM-IMPLEMENTATION.md`(把 DINOv3+SigLIP 编码器搬进 vLLM 的主体实现)、`MiniCPM-RobotTrack-Encoder-vLLM-TODO.md`(B-1/B-2 方案规划)。
> 本文汇总**「在 vision tower 内加逐帧特征缓存,让滚动窗口每步只重编码新帧」**这一步的**方案决策、实现改动、验证结果、运行方式、已知限制**。
> 状态:**已实现并端到端验证(GPU)**。日期:2026-08-04。

---

## 1. 目标与结论

**问题**:pixels-in 路径此前是 **B-1a(整窗口 = 一个 mm item)**,每步把整个 32 帧窗口重新过 DINOv3+SigLIP。但滚动窗口相邻两步有 31 帧完全相同 —— 这些历史帧被反复重编码,是纯浪费。

**目标**:让 tower 对**同一帧只编码一次**,历史帧在窗口内复用已编码特征;滚动请求每步只编码**新进入的当前帧**。

**结论**:实现完成。采用 **tower 内「按帧内容寻址」的特征缓存**(对齐上游 `vision_cache.py` 的做法),而非 vLLM 框架级 `encoder_cache` 的 B-2 改造。GPU 实测(50 帧,fp32/eager):稳态时延 **1056ms/步 → 409ms/步(2.58×)**,总时延 **2.52×**;缓存开/关输出一致性在 fp32 批量非确定性范围内(step0 逐值 `0.0`,中位 1.7e-3,最大 1.6e-2)。

**后续叠加「CPU 侧归一化像素缓存」(本文 §5.9)**:特征缓存消除的是 GPU 重编码,但每步 CPU 仍对整窗口 32 帧重做 resize+归一化(曾占 fp32 缓存开稳态的 ~107/409ms)。新增按帧内容寻址的归一化像素缓存后,CPU 每步只归一化新帧。RTX 4090 fp32/eager 实测(50 帧):稳态 **372.9ms/步 → 222.8ms/步(1.67×)**,总时延 14.23s → 9.61s;像素缓存开/关轨迹**逐位一致(max|Δtraj|=0.0)**。再叠加 fp16:稳态 **214.2ms/步**。至此每步 CPU mm(`_call_hf_processor`)从 ~150ms 降到 ~55ms(余项为 32 帧内容哈希 + 像素张量拼装 + tokenize)。

---

## 2. 方案决策:为什么用 tower 内缓存,而不是框架级 B-2

调研后**放弃 B-2(每帧一 mm item + 框架 encoder_cache)**,选 tower 内缓存。记录原因以免重复踩坑:

| 维度 | B-2(框架 encoder_cache) | **tower 内容缓存(采用)** |
|---|---|---|
| 每帧一生编码次数 | **2×**(缓存的是 pool 后、依赖 role 的结果) | **1×**(缓存 role-无关的 encode) |
| 时序 marker | 需作独立 item **与帧 item 交错** | 沿用现有 `_embed_visual_bundle` 自然拼装 |
| 代码改动 | 大且脆弱(与 vLLM 抽象冲突) | 局部(~`_encode_window` 一处) |
| 是否惯用 | 被 marker hack 抵消 | 对齐上游官方 `vision_cache.py` |
| 跨请求持久 | 是(至容量驱逐) | 是(挂在模型上,进程内常驻) |

关键调研发现(均已核对源码):

1. **框架 `encoder_cache` 对 pooling 路径确实跨请求生效**:编码输出存 `gpu_model_runner.encoder_cache`(`dict[str, Tensor]`),请求结束只移入 `freeable`、不立刻释放,后续同 `mm_hash` 命中即跳过 `embed_multimodal`(`encoder_cache_manager.py` / `scheduler.py:1466`)。**所以 B-2 技术上可行**——但代价见下。
2. **marker 是「相对窗口位置」**:上游 `data.py` 与离线示例均用 `arange(history_frames)`(0..30),**不是绝对帧时间**。同一物理帧随窗口滑动位置递减 → marker 变 → 不能折进「按内容缓存」的帧 item。
3. **vLLM 假设同一模态 item 同构**:`MultiModalKwargsItems.from_hf_inputs`(`inputs.py:952-962`)要求一个模态下所有字段等 batch size、**每个 item 拿到全部字段键**。因此「marker item 与 frame item 交错、字段不同」无法干净表达 → B-2 只能用**保留特殊 token id + 覆写 embedding** 的 hack。
4. **框架缓存的是 pool 后结果**(coarse=4 / fine=64,依赖 role)→ 同一帧作 fine(当前)与 coarse(历史)是两条缓存 → **2× 编码**。
5. **上游自带的正是 tower 缓存**:`~/MiniCPM-Robot/.../vision_cache.py: VisionFeatureCacher` 把每帧 pool 后的 coarse+fine 按帧身份缓存、每步重新拼装 —— 即本方案的思路,每帧只编码 1×(上游两池都缓存;本实现只缓存 coarse,fine 一次性使用,见 §5.2)。
6. **Reviewer 的 EXIF 方案(PR #49698)实际不触发**:`_parse_image_data → normalize_image`(`exif_transpose`)在哈希前就把 EXIF 抹掉了。

> DINOv3+SigLIP 的 fused grid 是 **role-无关**的:coarse/fine 一次前向即可 pool 出,故一帧只需编码一次。缓存只存 **coarse**(历史帧按窗口角色复用);fine 只属于当前帧、每次现场算——当前帧本就是刚进窗口的 miss,不产生额外编码。这是 tower 缓存优于框架缓存(2×)的根因。

---

## 3. 落地文件

| 文件 | 改动 |
|---|---|
| `vllm/transformers_utils/configs/minicpm_robottrack.py` | 新增配置项 `frame_cache_size`(默认 `64`;**`0` 关闭缓存 → 每步重编码,用作 A/B 对拍开关**)与 `pixel_cache_size`(默认 `64`;**`0` 关闭 CPU 像素缓存 → 每步全窗口重归一化,用作 A/B 对拍开关**)。 |
| `vllm/model_executor/models/minicpm_robottrack.py` | 新增 `_frame_content_key`、`_encode_frames_cached`、`_pixel_window_cached`、`_split_by_lengths`;改 `_call_hf_processor` / `_get_mm_fields_config` / `_encode_window` / `_embed_pixel_windows` / `embed_multimodal` / `__init__` / `MiniCPMRobotTrackProcessingInfo.__init__` / `prepare_pixels` / DummyInputsBuilder(详见 §5)。 |
| `tests/models/multimodal/pooling/test_minicpm_robottrack.py` | 新增 4 个免-GPU 帧缓存单测(`test_frame_cache_*`)+ 5 个免-GPU 像素缓存单测(`test_pixel_cache_*`)。 |

---

## 4. 数据流(pixels-in + 缓存)

```
客户端 / 会话层(状态仍在客户端):
  deque(maxlen=32) 存原始帧;每步发整窗口原始帧
     mm_data={"image": {"frames": [<=32 帧 HxWx3]}}

vLLM(模型内):
  _call_hf_processor(CPU,逐请求):
    tokenize(instruction)
    prepare_pixels(window):frame_keys[F]=blake2b(每帧字节)  # 内容寻址键
      _pixel_window_cached(frames, keys, self._pixel_cache, size, ...):
        miss=[不在像素缓存的帧];  只对 miss 帧 resize384+DINO/SigLIP 归一化
        其余帧直接取缓存归一化像素                       # ← 省掉 CPU 重归一化
      keys 与像素缓存同键,返回给调用方(frame_keys 只 hash 一次)
  embed_multimodal → _embed_pixel_windows → _encode_window:
    _encode_frames_cached(tower, self._frame_cache, size, dino, siglip, keys, 4, 64):
       miss = [不在特征缓存的帧];  只对 miss 帧过 tower(index_select 成子批)
       对 miss 帧 grid_pool 出 coarse(4)+fine(64);coarse 入 LRU,fine 供当前帧用后即弃
       其余帧直接取缓存                              # ← 省掉 GPU 重编码
    组装:末帧 fine(64);前 F-1 帧 coarse,_pad_history_frames 到 31
    造相对 time_indices(coarse 0..30 / fine 31)
    _embed_visual_bundle(projector + 逐帧 marker + control_query)→ [221, hidden]
  backbone(MiniCPMModel) + LAST pooling(control_query)+ trajectory head → [24]
```

**与改动前的关系**:池化/拼装/marker/backbone/head **完全不变**;唯一变化是 (a) `_encode_window` 内「整窗口过 tower」改为「只对 cache-miss 帧过 tower + 复用缓存」,以及 (b) `prepare_pixels` 内「整窗口 resize+归一化」改为「只对像素缓存 miss 帧归一化 + 复用缓存」。`frame_cache_size=0` / `pixel_cache_size=0` 时分别与旧路径逐值等价(像素缓存逐位等价,见 §6.4;特征缓存因批量归约非确定性在 fp32 批量差异范围内)。

---

## 5. 关键实现点(file: 符号)

`vllm/model_executor/models/minicpm_robottrack.py`:

1. **`_frame_content_key(frame) -> int`**(模块级):`blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8)` → int64。字节相同的帧(滚动时同一物理帧被反复下发)→ 同键 → 只编码一次。

2. **`_encode_frames_cached(tower, cache, cache_size, dino_pixels, siglip_pixels, frame_keys, coarse_tokens, fine_tokens)`**(模块级、可单测):
   - 命中缓存的帧直接取 `coarse` 并 `move_to_end`;未命中收集为 `miss_idx`。
   - 仅对 `miss_idx` 帧 `index_select` 成子批过 `tower`,`_grid_pool` 出 coarse+fine;**coarse 写入 LRU**(超 `cache_size` 则 `popitem(last=False)`)。fine 只属于当前帧、现场算不缓存(若当前帧是 coarse 命中,也仅对该单帧重跑 tower 取 fine)。
   - 返回 `(coarse_by_frame, fine, num_encoded)`(fine 为当前帧单个张量)。**`cache_size==0` 时不读不写 → 每帧必编码 = 旧路径**。

3. **`_call_hf_processor`**:逐窗口 `prepare_pixels` 的同时 `keys.extend(_frame_content_key(f) ...)`,输出 `frame_keys`(int64,`torch.long`)。

4. **`_get_mm_fields_config`**:新增 `frame_keys=MultiModalFieldConfig.flat_from_sizes("image", frame_lengths)`(与 dino/siglip 同按帧切分)。

5. **`MiniCPMRobotTrackModel.__init__`**:`self._frame_cache: OrderedDict[int, Tensor]`(仅 coarse)、`self._frame_cache_size = config.frame_cache_size`。

6. **`_encode_window(dino, siglip, frame_keys)`**:调 `_encode_frames_cached` → 取末帧 fine、前 F-1 帧 coarse → `_pad_history_frames` 到 31 → 造相对 time → `logger.debug` 打印命中情况。

7. **`_embed_pixel_windows` / `embed_multimodal`**:按 `frame_lengths` 用 `_split_by_lengths` 把 `frame_keys` 切到每个窗口并透传。

8. **`MiniCPMRobotTrackDummyInputsBuilder.get_dummy_mm_data`**:dummy 帧改为**逐帧不同填充值**(`np.full(..., i % 256)`),否则全零帧同键 → 缓存合并 → profiling 只编码 1 帧、低估 tower 峰值显存。

9. **`MiniCPMRobotTrackProcessingInfo`(CPU 像素缓存,§5.9 后续新增)**:
   - **`__init__`**:`self._pixel_cache: OrderedDict[int, tuple[Tensor, Tensor]]`(每帧 `(dino_pixel[1,3,H,W], siglip_pixel[1,3,H,W])`,fp32 CPU)、`self._pixel_cache_size = config.pixel_cache_size`。挂在 processor 持有的 info 单例上(renderer 单例,跨请求存活),与模型侧 `_frame_cache` 相互独立。
   - **`prepare_pixels(frames) -> (dino, siglip, keys)`**:先 `keys=[_frame_content_key(f)]`(blake2b-8,与特征缓存同键),再调 **`_pixel_window_cached`**:命中帧直接取缓存归一化像素并 `move_to_end`;miss 帧(含首帧,逐帧内容寻址)才 `_resize_frame`(PIL BICUBIC)+ 两个 HF image processor 归一化,结果入 LRU(超 `pixel_cache_size` 则 `popitem(last=False)`)。`pixel_cache_size=0` 时不读不写 → 全窗口归一化 = 旧路径。**返回 keys 供 `_call_hf_processor` 直接用,帧只 hash 一次。**
   - **`_pixel_window_cached(frames, keys, cache, cache_size, process_misses)`**(模块级、可单测):纯逻辑,`process_misses(miss_frames) -> (dino, siglip)` 注入真实的 resize+归一化;返回整窗口顺序的 `(dino, siglip)`。
   - **`_call_hf_processor`**:`dino_pixels, siglip_pixels, window_keys = self.info.prepare_pixels(window)`,删去原先的 `_frame_content_key` 二次调用。

   一致性根因:HF image processor 的逐帧归一化是逐图元素级运算,**批大小不影响单帧结果**,故像素缓存命中帧与重归一化帧**逐位相同** → §6.4 实测 max|Δtraj|=0.0(优于特征缓存的 fp32 批量非确定性)。

---

## 6. 验证

### 6.1 免-GPU 单测(缓存逻辑)

`tests/models/multimodal/pooling/test_minicpm_robottrack.py`,用计数假 tower(`_CountingTower`)驱动滚动窗口:

| 测试 | 断言 | 结果 |
|---|---|---|
| `test_frame_cache_encodes_each_frame_once` | 稳态每步只编码 1 帧;全程编码次数 = 不同帧数 | ✅ |
| `test_frame_cache_disabled_reencodes_full_window` | `cache_size=0` 时每步编码 = 窗口大小 | ✅ |
| `test_frame_cache_evicts_oldest_beyond_capacity` | 容量小于窗口时缓存不超上限 | ✅ |
| `test_frame_cache_matches_uncached_features` | 缓存开/关逐帧 coarse、当前帧 fine `allclose` | ✅ |
| `test_frame_cache_stores_coarse_only` | 缓存值仅为 coarse 张量(4×4),非 `(coarse, fine)` 元组 | ✅ |
| `test_frame_cache_reencodes_repeated_current_frame` | 重复帧作为当前帧时对其单帧重跑 tower 取 fine | ✅ |

全套 34 用例:32 通过 + 2 个 e2e 门控(需本地 checkpoint 环境变量)。

### 6.2 端到端时延 + 一致性(GPU,fp32)

环境:GPU 0,`runner="pooling"`,fp32,`enforce_eager`,`track-image/0` 前 50 帧,`frame_cache_size` 分别 64 / 0。脚本见 §7。

**时延(每步 `llm.embed`,`cuda.synchronize` 计时):**

| | step0 | 稳态均值(≥32 步) | 总时延 |
|---|---|---|---|
| 缓存**关**(`=0`) | 189ms | **1056ms/步** | 36.52s |
| 缓存**开**(`=64`) | 181ms | **409ms/步** | 14.46s |

- **稳态加速 2.58×,总加速 2.52×。**
- 逐步时延曲线(每 5 步)清晰印证机制:

```
step:     0     5    10    15    20    25    30    35    40    45
OFF (ms):189   213   348   503   675   832  1014  1057  1051  1053   ← 随窗口线性上升到满窗 32 帧编码成本
ON  (ms):181   108   135   185   227   309   399   413   403   402   ← 预热填缓存后稳态平台
```

**输出一致性(缓存开 vs 关,同 50 帧):**

| 指标 | 值 |
|---|---|
| step0 逐值差 | **`0.0`**(窗口仅 1 帧,两路都是 batch-of-1 → 逐位相同,证明缓存逻辑正确) |
| 逐步最大差中位数 | 1.7e-3 |
| 全程最大差 | 1.6e-2(step 23) |
| 轨迹值量级 | [0, 0.586] |

差异**非单调、不随步累积**,源自 fp32 下**批大小不同导致的 GPU 归约非确定性**(缓存路径历史帧来自 batch-of-1 编码,重编码路径来自 batch-of-F),与既有 pixels-in vs features-in 对拍(4.2e-3)同源同量级,非逻辑 bug。

### 6.3 端到端时延 + 一致性(GPU,fp16)

同一环境改 `dtype="float16"`(仅此不同),其余同 §6.2。

**时延:**

| | step0 | 稳态均值(≥32 步) | 总时延 |
|---|---|---|---|
| 缓存**关**(`=0`) | 880ms | **524ms/步** | 20.57s |
| 缓存**开**(`=64`) | 822ms | **408ms/步** | 15.73s |

- **稳态加速 1.28×,总加速 1.31×**(明显低于 fp32 的 2.58×)。
- 逐步时延曲线(每 5 步):

```
step:     0     5    10    15    20    25    30    35    40    45
OFF (ms):880   173   215   278   374   537   503   523   523   522
ON  (ms):822   140   132   264   228   317   450   456   465   378
```

**输出一致性(缓存开 vs 关,同 50 帧):** step0 逐值差 **`0.0`**;逐步最大差中位 1.5e-3,全程最大 9.3e-3(step 30),量级 [0, 0.585]。与 fp32 同源(批量非确定性),fp16 下反而略小。

**为什么 fp16 加速比骤降(重要发现):** 不是缓存失效,而是 **fp16 让 GPU 编码本身变便宜,缓存能省下的绝对 GPU 时间随之缩水,于是每步固定的 CPU 归一化开销(§8「归一化仍每步全窗口重算」)成为主导项**:

- **缓存开的稳态几乎不随精度变**:fp32 409ms → fp16 408ms。缓存开时每步只编码 1 帧,时间大头在**每步全窗口 resize/归一化(CPU)**与 backbone,这些不吃 fp16 红利。
- **缓存关大幅变快**:fp32 1056ms → fp16 524ms。fp16 把「重编码 32 帧」这个 GPU 大头砍半,故缓存可省的量变小,加速比从 2.58× 降到 1.28×。

> 结论:缓存收益随编码精度下降而收窄(fp32 2.58× → fp16 1.28×,仍为正)。**要在 fp16 下进一步提速,下一步应缓存「归一化后的像素」而非再优化编码**——即 §8 第 2 条已点名的瓶颈,在 fp16 下已升为主导。

### 6.4 端到端时延 + 一致性(CPU 像素缓存,G 并行)

同一环境(RTX 4090,`track-image/0` 前 50 帧),`frame_cache_size=64` 固定,`pixel_cache_size` 分别 0 / 64:

**时延:**

| | step0 | 稳态均值(≥32 步) | 总时延 |
|---|---|---|---|
| 像素缓存**关**(`=0`) | 1156ms | **372.9ms/步** | 14.23s |
| 像素缓存**开**(`=64`) | 1129ms | **222.8ms/步** | 9.61s |
| 像素缓存**开**+**fp16** | 1086ms | **214.2ms/步** | 8.36s |

- **稳态加速 1.67×,总加速 1.48×**(fp32);叠加 fp16 后稳态再降 ~4%(fp16 红利有限,说明余下瓶颈已非 GPU 算力)。
- **输出一致性(像素缓存开 vs 关,同 50 帧):max|Δtraj| = `0.0`,mean = `0.0` —— 逐位一致**,优于特征缓存的 fp32 批量非确定性(§6.2)。根因:归一化是逐图元素级,缓存命中帧与重归一化帧逐位相同;GPU 路径输入像素一致 → 轨迹一致。

**CPU 侧记账(profile_v2,36 步):** 像素缓存开后 `_call_hf_processor`(CPU mm)均值 **55.5ms/步**,占总步长 ~25%。余项构成:32 帧内容哈希(blake2b,~13ms)+ 像素张量拼装(torch.cat 114MB,~10ms)+ 1 帧 resize/归一化(~8ms)+ tokenize。**CPU 路径的大头(整窗口 32 帧 resize+归一化 ~107ms)已消除。**

---

## 7. 运行方式

度量脚本 `verify_cache.py`(仓库根,临时工具,未纳入版本):

```bash
# 特征缓存 + 像素缓存 都开(默认)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python verify_cache.py \
  --model /cache/zhanghao/model/MiniCPM-RobotTrack \
  --dino  /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
  --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
  --images track-image/0 --frame-cache-size 64 --pixel-cache-size 64 --out /tmp/pix_on.npz

# 特征缓存关 / 像素缓存关(仅改对应 flag)
  ... --frame-cache-size 0 --out /tmp/cache_off.npz
  ... --pixel-cache-size 0 --out /tmp/pix_off.npz

# fp16 组:再加 --dtype float16(默认 float32)
  ... --dtype float16 --frame-cache-size 64 --pixel-cache-size 64 --out /tmp/pix_on_fp16.npz

# 一致性 + 加速比汇总(任意两组 .npz)
.venv/bin/python verify_cache.py --compare /tmp/pix_on.npz /tmp/pix_off.npz
```

正常业务入口不变(两缓存默认开,`frame_cache_size=64` / `pixel_cache_size=64`):

```python
llm = LLM(model=".../MiniCPM-RobotTrack", runner="pooling", dtype="float32",
          enforce_eager=True, enable_mm_embeds=True, limit_mm_per_prompt={"image": 1},
          hf_overrides={"dino_model": ".../dinov3-vits16-pretrain-lvd1689m",
                        "siglip_model": ".../siglip-so400m-patch14-384",
                        "image_size": 384})   # frame_cache_size / pixel_cache_size 可选,默认 64
```

免-GPU 单测:

```bash
.venv/bin/python -m pytest tests/models/multimodal/pooling/test_minicpm_robottrack.py -k "frame_cache or pixel_cache" -v
```

---

## 8. 有状态流协议(每步只发 1 帧)

**目标**:消除上表「框架/IPC/H2D ~118ms」——把窗口状态搬到 server,客户端每步只发 1 帧。**已实现并端到端验证。**

**请求契约**(`stream_id` 由客户端生成,`frame_index` 是单调帧号):

```
建立:   {"frames": [32 帧], "stream_id": sid, "frame_index": N}   # replace
推进:   {"frames": [1 帧],  "stream_id": sid, "frame_index": N+1} # append
重试:   {"frames": [1 帧],  "stream_id": sid, "frame_index": N}   # reuse(幂等)
```

**实现**(`vllm/model_executor/models/minicpm_robottrack.py`):
- **`RobotTrackStreamState`**(模块级 dataclass):`coarse_history`(31 帧 coarse 池,最新在后,不含当前帧)+ `current_coarse`(当前帧 coarse,append 时提升进历史)+ `fine` + `frame_index`。
- **`_classify_stream_request`**(纯逻辑,单测):32 帧 → `replace`;1 帧 + 无流 → 报错「先建立」;`frame_index == prev` → `reuse`;`== prev+1` → `append`;否则报错「乱序」。
- **`_encode_stream_window`**:按 mode 编码/复用,`_advance_stream_state` 滚动窗口,**policy 前 commit**(失败重试靠 `reuse`/`append` 二义消除)。`_assemble_stream_window` 与无状态 `_encode_window` 产出完全相同的 `(coarse, coarse_time, fine, fine_time)`。
- **`_embed_stream_windows`**:批字段 `stream_id`/`frame_index` 经 H2D 后是 0-dim CUDA tensor,先 `_scalar` 转回 int 再作 dict 键(tensor 按身份 hash,直接当键永远匹配不上)。
- **parser/items**:`{"frames", "stream_id", "frame_index"}` → `MiniCPMRobotTrackPixelItems` 携带;`stream_id` 字符串经 `_stream_id_key`(blake2b-8)转稳定 int 键(**vLLM IPC 序列化器 `_encode_nested_tensors` 只支持 Tensor/int/float,字符串会无限递归**);cache-miss 重解析会把 item 包成 `[dict]`,parser 需解包保留元数据。
- 配置 `max_cached_streams`(默认 8,LRU)。

**验证**(RTX 4090,50 帧):有状态 vs 无状态整窗**轨迹逐位一致至 fp32 非确定性(max|Δ|=1.8e-4);稳态单帧 **58.5ms/步**(vs 无状态整窗 223ms,**3.8×**)。示例:`examples/pooling/robottrack_minicpm_stream.py`;对照:`verify_stream.py`。

**已知限制**:流状态是模型内显式会话状态——DP>1 需亲和路由或 DP=1;`reuse` 后 client 需在失败时重发同一 `frame_index`;协议错误经 `llm.embed` 异常(带错误码消息)返回,客户端 catch 后重建。

---

## 9. 已知限制 / 下一步

- **缓存为模型内状态**:已实测 **CUDA graph 可用**(`embed_multimodal` 在捕获区外的 mm-encoder 路径执行,缓存读写不受捕获影响;40 步实测输出有限/24 维,稳态 ~309ms,与 eager 缓存开同量级)。TP 下各 rank 各持一份内容一致的缓存(重复占显存)仍待评估。
- **CPU 像素缓存已落地(§5.9)**:原先「归一化仍每步全窗口重算」的瓶颈已消除(整窗口 32 帧 resize+归一化 ~107ms → 每步 1 帧 ~8ms)。像素缓存挂在 processor 的 info 单例上,与模型侧特征缓存同键、相互独立;`pixel_cache_size=0` 可逐位回退旧路径。
- **剩余 CPU(约 55ms/步)**:`_call_hf_processor` 里仍是每步对 32 帧做 blake2b 内容哈希(~13ms)与 114MB 像素张量 `torch.cat` 拼装(~10ms),外加 tokenize。要再降需改请求契约(客户端带 frame id,省掉哈希)或缓存整窗口拼装结果。
- **余下稳态 ~223ms 的精确构成**(in-engine CUDA event + client `processor.apply` 计时):
  | 部分 | 时延 |
  |---|---|
  | CPU mm(`_call_hf_processor` + `from_hf_inputs` + mm-hash)| ~57ms |
  | 视觉塔 eager 编码 1 帧 + pool(DINO 13.6 + SigLIP 22.6)| ~33ms |
  | bundle 拼装(projector+marker+cat)| ~7ms |
  | backbone(torch.compile)| ~7ms |
  | **框架 / IPC / H2D(每步跨进程传 ~114MB 归一化像素到 worker)**| **~118ms** |

  关键结论:**视觉塔不是大头**(eager 只 ~33ms),fp16 也只再降 ~4%。最大头是 ~118ms 的**每步跨 IPC 传递 114MB mm payload**(client mm 处理完 → 序列化 → engine core → worker H2D)。**该瓶颈已由 §8 的有状态流协议解决**(每步只发 1 帧 → 稳态 223ms → 58.5ms)。
- **缓存显存不计入框架 budget**:特征缓存由 `frame_cache_size` 限界(仅 coarse:64 × 约 24KB ≈ 1.5MB);像素缓存由 `pixel_cache_size` 限界(每帧 dino+siglip fp32 ≈ 3.5MB,64 帧 ≈ 226MB CPU)。单调视频中帧离窗后不再返回,LRU 64 足够。
- **PR / 人工问责**:按 `AGENTS.md`,AI 辅助工作需人工逐行 review、跑测试;PR 描述须含非重复性 / 测试命令与结果 / 模型评测 / AI 辅助声明。

---

## 9. 版本 / 环境备注

- 主环境:本仓库 `.venv`,torch 2.11.0+cu130;GPU 单卡 fp32/eager。
- 配置默认(`MiniCPMRobotTrackConfig`):`history_frames=31`、`coarse_tokens_per_frame=4`、`fine_tokens_current_frame=64`、`image_size=384`、`frame_cache_size=64`。
- 编码器权重(DINOv3-S / SigLIP-so400m-384)仍经 `hf_overrides` 外挂,不在 RobotTrack checkpoint 内(见编码器实现报告)。

---

## 10. PR 回复参考(英文,可直接贴到 PR / issue 讨论)

> 以下是可直接用于 PR 讨论的英文回复草稿。核心是主动摊开「模型自持缓存」这一取舍,请 reviewer 定夺方向,而非回避。请人工核对后再发。

---

**Follow-up on the RobotTrack sliding-window re-encoding / caching discussion (re: #49698)**

**Context.** RobotTrack runs a rolling 32-frame window; between consecutive
`embed` requests, 31 of the 32 frames are identical, and the current pixels-in
tower re-encodes the whole window every step. This PR addresses that.

**What I implemented.** A per-frame, content-addressed feature cache *inside the
model* (`DualVisionTower` path). Each raw frame is hashed
(`blake2b(frame_bytes)`) and its role-independent `coarse` pooled features
(`coarse=4`, from the fused DINOv3+SigLIP grid) are memoized in a bounded LRU on
the model instance. The current frame's `fine=64` pool is never cached: it is
consumed only once (while that frame is current) and recomputed fresh each step,
which is normally a cache miss anyway. A rolling request therefore encodes only
the newly-arrived frame; the ~31 unchanged history frames are served from cache.
Temporal markers / control-query assembly are unchanged (still rebuilt per step,
since marker position is **relative** to the window). A config flag
`frame_cache_size` bounds it, and `frame_cache_size=0` fully disables reuse
(exact re-encode) — used as the A/B toggle below.

**Why a model-held cache rather than the framework `encoder_cache` (the
"one-item-per-frame" restructure).** I prototyped that direction first and
stepped back for three concrete reasons:
1. **Relative markers.** Marker embeddings depend on a frame's *position in the
   window* (`arange(history_frames)`), which changes as the frame ages. Folding
   markers into a per-frame item makes that item's hash change every step →
   cache never hits. Keeping markers as separate interleaved items is possible
   only via reserved special-token ids + embedding overrides.
2. **Homogeneous-item assumption.** `MultiModalKwargsItems.from_hf_inputs`
   requires every item in a modality to carry the same field keys with equal
   batch sizes, so interleaving heterogeneous marker-items and frame-items in one
   modality isn't clean.
3. **The framework cache stores *pooled* (role-dependent) outputs**, so the same
   physical frame is encoded twice over its lifetime (once as `fine` current,
   once as `coarse` history). The role-independent fused grid is what's actually
   expensive, and a content cache encodes each frame exactly **once**.
   This also matches the upstream reference (`vision_cache.py`
   `VisionFeatureCacher`), which caches per-frame pooled features by frame
   identity.

**Test results.**
- *Unit (no GPU):* 6 tests with a counting fake tower assert (a) steady state
  encodes exactly 1 frame/step, (b) each distinct frame is encoded once over the
  run, (c) LRU bound is respected, (d) cache-on features == cache-off features
  (coarse per frame + the current frame's fine), (e) the cache stores coarse-only
  tensors (never `(coarse, fine)` tuples), (f) a frame re-appearing as the current
  frame re-encodes that single frame for its fine.
- *End-to-end (GPU, eager, 50 frames of a real clip), steady-state ms/step:*

  | dtype | cache off (`=0`) | cache on (`=64`) | speedup |
  |---|---|---|---|
  | fp32 | 1056 ms | **409 ms** | **2.58×** |
  | fp16 | 524 ms | **408 ms** | **1.28×** |

- *Precision-dependent speedup (worth flagging).* The cached steady state is
  essentially dtype-invariant (~408 ms) because with the cache on, each step
  encodes only 1 frame and the per-step cost is dominated by CPU
  resize/normalize of the full window + backbone, neither of which benefits from
  fp16. fp16 roughly halves the *uncached* full-window encode (1056→524 ms), so
  the absolute GPU time the cache saves shrinks and the ratio drops. Net: the
  cache is a clear win at fp32; still positive but smaller at fp16, where the
  per-step normalization becomes the dominant cost (see open question 4).
- *Parity (cache on vs off, same 50 frames):* step 0 is bit-identical (`0.0`;
  window is a single frame, so both encode a batch-of-1). Later steps differ by a
  median of 1.7e-3 / max 1.6e-2 (fp32) and 1.5e-3 / 9.3e-3 (fp16) on a value
  scale of [0, 0.59] — this is batch-size non-determinism (history features come
  from batch-of-1 encodes when cached vs batch-of-N when re-encoding),
  non-accumulating, and the same order as the pre-existing pixels-in vs
  features-in parity (4.2e-3).

**What I'd like reviewers to weigh in on.**
1. **Is a model-held cache acceptable here, or should this go through the
   framework `encoder_cache`?** It's the crux. The model-held approach is simple,
   matches upstream's own reference, and encodes each frame once — but it does
   introduce model-instance state, which the framework path avoids.
2. **CUDA graph / TP interaction.** The cache is validated under single-worker
   eager. Under graph capture the read/write is a side effect; under TP each rank
   holds its own copy (correct but duplicated, and not counted against the
   framework encoder budget). Do we want to gate `frame_cache_size>0` to
   eager/TP=1, or is documenting the limitation enough?
3. **Cache key.** Currently `blake2b` over raw frame bytes. Is a content hash the
   right identity, or should this key off a client-supplied frame id?
4. **Remaining cost / precision.** The window item's `mm_hash` still changes every
   step, so CPU resize/normalize of the full window is not cached (the expensive
   GPU encode is). At fp16 this normalization is already the dominant per-step
   cost (see above). Worth caching normalized pixels too, or out of scope for this
   PR?

*AI assistance was used for this change; all lines were human-reviewed and the
tests above were run locally.*
