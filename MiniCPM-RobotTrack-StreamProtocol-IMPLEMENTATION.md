# MiniCPM-RobotTrack 有状态流协议 → vLLM 实现报告

> 配套文档:`MiniCPM-RobotTrack-FrameCache-IMPLEMENTATION.md`(GPU 特征缓存 + CPU 像素缓存,§1-§7)、`MiniCPM-RobotTrack-Encoder-vLLM-IMPLEMENTATION.md`(把 DINOv3+SigLIP 编码器搬进 vLLM 的主体实现)。
> 本文汇总**「把滚动窗口搬到 server,客户端每步只发 1 帧」**这一步的**方案决策、请求契约、代码改动、验证结果、踩坑记录、已知限制**。
> 状态:**已实现并端到端验证(GPU)**。日期:2026-08-06。

---

## 1. 目标与结论

**问题**:无状态像素路径每步把整窗口(32 帧,114MB 归一化像素)跨 IPC 传给 worker,稳态 223ms/步里 ~118ms 是框架/IPC/H2D、~55ms 是 CPU mm——都押在「每步搬 32 帧」上(见 FrameCache 报告 §9 的精确构成)。

**目标**:把窗口状态搬到 server(按 `stream_id`),客户端每步只发 **1 帧** + 单调 `frame_index`,消除 114MB payload 与 32 帧 CPU 处理。

**结论**:实现完成。RTX 4090 实测(50 帧):有状态稳态 **57.4ms/步**(vs 无状态整窗 223ms,**3.9×**);有状态 vs 无状态同一窗口轨迹 **max|Δ| = 1.8e-4**(fp32 批量非确定性);41 个单测通过、lint 干净。

---

## 2. 方案背景:为什么转有状态

FrameCache 报告已把稳态 223ms 拆开(in-engine CUDA event + client `processor.apply` 计时):

| 部分 | 时延 |
|---|---|
| CPU mm(`_call_hf_processor` + `from_hf_inputs` + mm-hash) | ~57ms |
| 视觉塔 eager 编码 1 帧 + pool | ~33ms |
| bundle 拼装 | ~7ms |
| backbone(torch.compile) | ~7ms |
| **框架 / IPC / H2D(每步跨进程传 ~114MB)** | **~118ms** |

最大头是 **每步跨 IPC 搬 114MB**(client mm 处理完 → 序列化 → engine core → worker H2D)。特征缓存 / 像素缓存都只是省了「重算」,没省「搬运」。把窗口状态搬进 server 后,每步只跨 IPC 传 1 帧(≈3.5MB)+ 31 帧不用传——~118ms IPC 和大部分 ~57ms CPU mm 一起消掉。

与「发 key + 新帧」折中方案的取舍:有状态方案让 server 记会话(而非客户端算 key / 回显 server 签发 key),换来客户端零 hash 负担、无 key 驱逐握手;代价是 server 端会话生命周期与 DP 受限(见 §9)。PhyAI PR #51 正是这个做法(stream_id + frame_index + LRU)。

---

## 3. 请求契约

`stream_id` 由客户端生成(字符串或整数,字符串被 `_stream_id_key` 哈希成稳定 int);`frame_index` 是客户端单调帧号(相机帧计数器即可,不必映射到时间)。

```
建立:  {"frames": [32 帧],   "stream_id": sid, "frame_index": N}   # replace
推进:  {"frames": [1 帧],    "stream_id": sid, "frame_index": N+1} # append
重试:  {"frames": [1 帧],    "stream_id": sid, "frame_index": N}   # reuse(幂等)
```

| 输入 | 校验 | server 动作 |
|---|---|---|
| 32 帧,stream 不存在 | frame_index ≥ 0 | **replace**:编码全窗,建立状态 |
| 32 帧,stream 存在 | — | **replace**:替换整窗(重新同步) |
| 1 帧,stream 不存在 | — | 报错「先发 32 帧建立」 |
| 1 帧,`frame_index == prev` | — | **reuse**:不编码,用已提交状态(幂等重试) |
| 1 帧,`frame_index == prev+1` | — | **append**:编码新帧,滚动窗口 |
| 1 帧,`frame_index` 跳跃/回退 | — | 报错「乱序」 |
| stream 被 LRU 淘汰 | — | 报错「重新建立」 |

**幂等键 = `(stream_id, frame_index)`**:客户端请求失败后不知道 server 是否已提交,重发**同一个** `frame_index`——server 用 reuse(已提交)/ append(未提交)二义消除。提交时机在 `_encode_stream_window` 内 encode+组装成功后(policy 前);若 policy 失败,重试走 reuse 重建,结果一致。

---

## 4. 代码改动清单

| 文件 | 位置 | 改动 |
|---|---|---|
| `vllm/transformers_utils/configs/minicpm_robottrack.py` | `__init__` 签名 / `max_cached_streams` 字段 | 新增 `max_cached_streams: int = 8`(流状态 LRU 上限) |
| `vllm/model_executor/models/minicpm_robottrack.py` | 见 §5 | 流状态 + 透传 + 分发(下述) |
| `tests/models/multimodal/pooling/test_minicpm_robottrack.py` | §6.1 | 4 个流测试 + 既有像素/特征缓存测试 |
| `examples/pooling/robottrack_minicpm_stream.py` | — | 客户端示例:establish 32 帧 + 单帧推进 |
| `verify_stream.py` | — | 对照验证:有状态 vs 无状态同一窗口 |

未改 vLLM 核心(`serial_utils.py` / `parse.py` / `processor.py`)——所有适配都在模型侧完成,靠 §6.2 的框架约束规避。

---

## 5. 关键实现点(`vllm/model_executor/models/minicpm_robottrack.py`)

### 5.1 纯逻辑模块(可单测)

- **`_stream_id_key(stream_id) -> int`**(`:77`):字符串经 `blake2b-8` 哈希成稳定 int。原因见 §6.2-坑2。
- **`RobotTrackStreamState`**(`:136`):`coarse_history: tuple[Tensor,...]`(31 帧 coarse 池,最新在后,**不含当前帧**)+ `current_coarse`(当前帧 coarse,append 时提升进历史)+ `fine`(当前帧 fine)+ `frame_index`。
- **`_classify_stream_request(frame_count, history_frames, state, frame_index) -> "replace"|"append"|"reuse"`**(`:152`):§3 状态机,纯逻辑、无 GPU。
- **`_advance_stream_state(state, mode, coarse_by_frame, fine, frame_index, history_frames)`**(`:195`):
  - `replace`:`history = coarse_by_frame[:-1][-31:]`,`current_coarse = coarse_by_frame[-1]`。
  - `append`:`history = (*state.coarse_history[1:], state.current_coarse)[-31:]`,`current_coarse = coarse[0]`。
  - **关键**:append 必须把「旧当前帧」的 coarse 提升进历史,否则窗口会跳帧。
- **`_assemble_window_tensors(coarse_history, fine, ...) -> (coarse, coarse_time, fine, fine_time)`**(`:220`):`_pad_history_frames(stack(coarse_history), 31)` + `arange(31).repeat_interleave(4)` + `full(64, 31)`——无状态 `_encode_window` 与有状态 `_assemble_stream_window`(`:246`,只 re-stack 状态里已提交的 coarse_history)共用,**parity 由同一段代码保证**而非两份拷贝手工同步。

### 5.2 模型侧

- **`MiniCPMRobotTrackModel.__init__`**(`:1341`):`self._stream_states: OrderedDict[int, RobotTrackStreamState]`、`self._max_cached_streams = config.max_cached_streams`。
- **`_commit_stream`**(`:1486`):LRU 写入,超上限 `popitem(last=False)`。
- **`_encode_stream_window(dino, siglip, frame_keys, stream_id, frame_index) -> bundle`**(`:1492`):classify → `reuse` 直接用已提交状态;否则 `_encode_frames_cached`(复用特征缓存)编码新帧 → `_advance_stream_state` → `_commit_stream` → `_assemble_stream_window` → `_embed_visual_bundle`。
- **`_embed_pixel_windows`**(`:1557`):批字段转标量(`_scalar`,见坑1),按 `frame_lengths` 切分逐窗口处理;`stream_id` 可选——有则走 `_encode_stream_window`(有状态),无则走 `_encode_window`(无状态,向后兼容)。
- **`embed_multimodal`**(`:1601`):`dino_pixels` 存在 → `_embed_pixel_windows`(stream_id 可选,一条路径);否则走 features-in 路径。

### 5.3 透传链(stream_id / frame_index 从 mm_data 到 embed_multimodal)

1. **`MiniCPMRobotTrackPixelItems`**(`:861`):构造器新增 `stream_id` / `frame_index`;`get_processor_data()` 带上它们;`get(0)` 在有状态时返回 `{"frames", "stream_id", "frame_index"}`(否则裸 list)——**为让框架 cache-miss 重解析保留元数据**,见坑3。
2. **`_parse_image_data`**(`:911`):识别 `{"frames", "stream_id", "frame_index"}`;**解包 cache-miss 重解析的 `[dict]` 单元素列表**(坑3);`stream_id` 经 `_stream_id_key` 转 int。
3. **`_call_hf_processor`**(`:1172`):`outputs["stream_id"] = [mm_data["stream_id"]]`、`outputs["frame_index"] = [mm_data["frame_index"]]`(1 元素列表)。
4. **`_get_mm_fields_config`**(`:1227`):`dino_pixels` 分支里,若 `hf_inputs` 有 `stream_id`,加 `stream_id` / `frame_index = MultiModalFieldConfig.batched("image")`(非张量值经 batched 字段逐 item 透传)。

---

## 6. 踩坑记录(框架约束,全部已解决)

1. **batched 字段 H2D 后变 0-dim CUDA tensor → 作 dict 键永远 miss**。`_embed_stream_windows` 里 `stream_id`/`frame_index` 到达时是 `tensor(32, device='cuda')`,而 tensor 的 `__hash__` 按对象身份,新请求的新 tensor 查不到旧 key。**修**:`_scalar(x) = int(x.item()) if isinstance(x, torch.Tensor)`。
2. **vLLM IPC 序列化器 `_encode_nested_tensors` 只支持 Tensor/int/float**——字符串会无限递归(`for x in "s0"` → 单字符字符串又迭代)。**修**:`stream_id` 字符串在 parser 里经 `_stream_id_key` 转 int,mm 字段只传 int。
3. **框架 cache-miss 重解析丢元数据**:`_get_cache_missing_items` 用 `mm_data_items[modality][idx]`(= `item.get(0)`)取 item 数据 → `parse_mm_data([data])` → `_parse_image_data([data])`。若 `get(0)` 返回裸 frames list,重解析出的 item 没有 stream_id;若返回 dict,`_parse_image_data([dict])` 会把 dict 当帧(→ PIL 报错)。**修**:`get(0)` 有状态时返回 dict,parser 解包 `[dict-with-frames]` 单元素列表。
4. **append 必须提升旧当前帧**:若状态只存 history + fine,append 时把旧 current 丢掉,窗口会跳一帧(初期 parity 0.25 的根因)。**修**:状态增加 `current_coarse` 字段,append 时 `history[1:] + current_coarse`。

---

## 7. 验证

### 7.1 免-GPU 单测

`tests/models/multimodal/pooling/test_minicpm_robottrack.py`:

| 测试 | 断言 | 结果 |
|---|---|---|
| `test_classify_stream_request_state_machine` | replace / append / reuse / 各错误分支 | ✅ |
| `test_stream_replace_keeps_last_31_history` | 32 帧 replace → history=31,current_coarse=第 32 帧 | ✅ |
| `test_stream_append_rolls_window` | append 提升旧 current、丢最旧、新帧成 current | ✅ |
| `test_stream_assemble_matches_stateless_window` | 40 帧滚动,组装恒为「末 31 coarse + 当前 fine」,与无状态对齐 | ✅ |

全套 41 通过 + 2 个 e2e 门控。

### 7.2 端到端(GPU,RTX 4090,50 帧)

`verify_stream.py`:同一进程跑有状态(establish + 单帧推进)与无状态(每步整窗)两条路径,对比同一窗口的轨迹。

| 指标 | 值 |
|---|---|
| 有状态稳态单帧 | **57.4ms/步**(p50 57.5) |
| 有状态 vs 无状态轨迹 | **max|Δ| = 1.8e-4**,mean 2.9e-5(量级 [-0.246, 0.586]) |
| 相对无状态整窗(223ms) | **3.9×** |

客户端示例 `robottrack_minicpm_stream.py`:establish(冷启动,含模型加载/编译)1280ms;稳态单帧 57.5ms。

### 7.3 一致性说明

有状态/无状态差异 1.8e-4 来自 fp32 批量非确定性(建立时 32 帧一起编码 vs 单帧编码的归约顺序不同),**非累积、逐位不保证但量级正确**,与既有特征缓存 parity(1e-3~1e-2)同源且更小。像素缓存与特征缓存在两条路径间共享、内容寻址,不影响。

---

## 8. 运行方式

```bash
# 客户端示例(establish 32 帧 + 单帧推进)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/pooling/robottrack_minicpm_stream.py \
    --model /cache/zhanghao/model/MiniCPM-RobotTrack \
    --dino /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
    --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
    --images track-image/0

# 对照验证(有状态 vs 无状态 parity + 时延)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python verify_stream.py \
    --model /cache/zhanghao/model/MiniCPM-RobotTrack \
    --dino /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
    --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
    --images track-image/0 --max-frames 50

# 免-GPU 单测
.venv/bin/python -m pytest tests/models/multimodal/pooling/test_minicpm_robottrack.py -v
```

客户端用法(协议):

```python
from vllm import LLM

llm = LLM(model=".../MiniCPM-RobotTrack", runner="pooling", dtype="float32",
          enable_mm_embeds=True, limit_mm_per_prompt={"image": 1},
          hf_overrides={"dino_model": "...", "siglip_model": "...", "image_size": 384})

def embed(frames, frame_index, stream_id="cam0"):
    mm = {"image": {"frames": frames, "stream_id": stream_id, "frame_index": frame_index}}
    return llm.embed([{"prompt": "Follow the person.", "multi_modal_data": mm}])[0].outputs.embedding

# 建立:32 帧
traj = embed(window_32, frame_index=31)
# 稳态:每步只发新 1 帧
for fi in range(32, 50):
    traj = embed([new_frame], frame_index=fi)      # 失败就重发同一 frame_index
```

---

## 9. 已知限制 / 下一步

- **显式会话状态**:`_stream_states` 在模型上,是模型内状态——与特征/像素缓存同列。**DP>1 需要会话亲和路由或 DP=1**(同一 stream 的请求必须落同一个 rank);TP>1 各 rank 持一致副本(显存重复,正确)。
- **错误语义**:协议错误(未建立 / 乱序 / 被淘汰)经 `llm.embed` 抛 `ValueError` 带消息返回,客户端 catch 后重发同一 `frame_index` 或重建 32 帧。要更干净需在 pooling 响应加状态字段(选项 B,动 vLLM 输出契约)。
- **`stream_id` 为 int 键**:字符串经 blake2b-8 哈希,理论上极小碰撞风险;客户端无需知道 int。
- **`current_coarse` 内存**:每流 31×4×1536 + 4×1536 ≈ 0.8MB,`max_cached_streams=8` ≈ 6.4MB。
- **下一步(压 57ms 内的大头)**:视觉塔 eager 编码仍占 ~33ms——mm-encoder CUDA graph + fp16/bf16 可再砍 ~15-25ms;或对 `embed_multimodal` 路径做 torch.compile(compile_mm_encoder)。
- **PR / 人工问责**:按 `AGENTS.md`,AI 辅助工作需人工逐行 review、跑测试;PR 描述须含非重复性 / 测试命令与结果 / 模型评测 / AI 辅助声明。

---

## 10. 版本 / 环境备注

- 主环境:本仓库 `.venv`,torch 2.11.0+cu130;GPU 单卡 fp32/eager;RTX 4090。
- 配置默认:`history_frames=31`、`coarse_tokens_per_frame=4`、`fine_tokens_current_frame=64`、`image_size=384`、`frame_cache_size=64`、`pixel_cache_size=64`、`max_cached_streams=8`。
- 相关实现:`FrameCache` 报告(特征+像素缓存)、`Encoder-vLLM` 报告(DINOv3+SigLIP in-tree)。
