# MiniCPM-RobotTrack 编码器 → vLLM 集成实现报告(初步版本)

> 配套文档:`MiniCPM-RobotTrack-Encoder-vLLM-TODO.md`(方案规划 + 模板参考)、
> `MiniCPM-RobotTrack-vLLM-IMPLEMENTATION.md`(方案 A / 特征-in 主体实现)。
> 本文汇总**「把 DINOv3+SigLIP 编码器搬进 vLLM」这一步的已实现内容、设计决策、验证结果、运行方式、已知限制**。
> 状态:**已完成并端到端验证**(真实图片跑通 + 编码器逐帧 parity)。日期:2026-07-23。

---

## 1. 目标与结论

把 MiniCPM-RobotTrack 的视觉编码器(**DINOv3 + SigLIP 融合**)从「vLLM 之外的客户端预处理」搬进 vLLM,做成**模型内的 vision tower**;**32 帧滚动窗口仍由客户端维护**。客户端每步只发**原始帧**,vLLM 内部完成 resize/归一化 + 编码 + 池化 + 时序拼装 + 策略前向,直接吐 8 路点。

**tower 完全 vLLM 原生**:两个编码器都是**in-tree vLLM 模块**,不再依赖 `transformers` 的编码器模型——SigLIP 复用 vLLM 原生 `SiglipVisionModel`,DINOv3 是新写的 in-tree 端口 `vllm/model_executor/models/dinov3.py`(vLLM 张量并行 linear + SDPA + 忠实 2D RoPE)。(`transformers` 仅用于:读 encoder config、CPU 侧 image processor 归一化——均非模型前向。)

**结论**:实现完成。同一批真实帧下,**pixels-in(vLLM 原生 tower)与 features-in(上游忠实 `DualVisionEncoder`)的 vLLM 24 维输出 `max abs diff = 4.2e-3`**(8 帧;逐帧 ~0.3–4e-3,为两次独立 fp32 编码器前向的数值噪声)。features-in 此前已对 HF golden 验证(`allclose(2e-3)`),故等价链成立:vLLM 原生 DINOv3 ≈ HF DINOv3,vLLM SigLIP ≈ HF SigLIP,均在 fp32 噪声内。DINOv3 RoPE 数学对 HF **逐值 `0.0`**。不依赖 `trust_remote_code`。

---

## 2. 采用的方案

| 维度 | 选择 | 说明 |
|---|---|---|
| item 切法 | **B-1a**:整窗口 = 一个 mm item | TODO 认定的「最快跑通」路线,契合「初步版本 + 客户端维护 deque」;非 B-2(每帧一 item)。 |
| 客户端契约 | **P1**:传原始帧 | `mm_data={"image": {"frames": [<=32 帧 HxWx3]}}`;processor 内跑 DINO/SigLIP image processor(resize384+各自归一化),tower 跑两个编码器。相较「客户端归一化后发像素」省 ~16× IPC。 |
| 缓存 | **无 tower 内缓存** | 每步重编码整窗口(DINOv3-S+SigLIP 不大;要省重编码应升级 B-2 命中框架 `encoder_cache`)。 |
| 兼容性 | **双模、向后兼容** | features-in(方案 A)路径完整保留;parser 按 dict 内容分派。 |

---

## 3. 落地文件

### 修改 / 新增
| 文件 | 改动 |
|---|---|
| `vllm/transformers_utils/configs/minicpm_robottrack.py` | 新增 `dino_model` / `siglip_model` / `image_size`(经 `hf_overrides` 注入本地路径;features-in 路径不用)。 |
| `vllm/model_executor/models/dinov3.py` | **新增**。in-tree 原生 DINOv3 ViT(vLLM 端口),供 tower 使用(见 §5.2)。 |
| `vllm/model_executor/models/minicpm_robottrack.py` | 新增 tower + pixels-in 管线(见 §4/§5);features-in 主体不动。 |
| `tests/models/multimodal/pooling/test_minicpm_robottrack.py` | pixels-in 单测 + 门控 e2e。 |
| `tests/models/multimodal/pooling/test_dinov3.py` | **新增**。DINOv3 RoPE / rotate_half 纯数学单测(无需引擎/TP)。 |
| `examples/pooling/robottrack_minicpm_video.py` | 客户端只维护 `deque(maxlen=32)` 原始帧、传窗口;`LLM(..., hf_overrides={dino_model, siglip_model, image_size})`。 |

> features-in 的离线 demo `examples/pooling/robottrack_minicpm_offline.py` 不受影响(仍走特征-in)。`transformers` 仍是依赖,但 tower 只用它读 config + 跑 image processor(CPU 归一化),**不再用它的编码器模型**。

---

## 4. 数据流(pixels-in)

```
客户端 / 会话层(状态仍在客户端):
  deque(maxlen=32) 存原始帧(PIL)
  每帧 -> 组 request: {prompt, mm_data={"image": {"frames": [<=32 帧 HxWx3 uint8]}}}
          (窗口:oldest first, current last;不足 31 历史由模型侧补齐)

vLLM(模型内):
  parser: dict 含 "frames" -> MiniCPMRobotTrackPixelItems(整窗口 = 1 个 item)
  _call_hf_processor(CPU,逐请求):
    tokenize(instruction)
    _normalize_windows(frames) -> 每个 item 一个窗口
    prepare_pixels(window): 每帧 RGB + BICUBIC resize384 -> DINO/SigLIP image processor 各自归一化
      -> dino_pixels[F,3,384,384], siglip_pixels[F,3,384,384], frame_lengths=[F]
  _get_prompt_updates: 末尾插 fixed 221 占位符(tower 内部把历史补到 31,故与帧数无关)
  embed_multimodal(逐 item):
    DualVisionTower(dino_pixels, siglip_pixels) -> 融合网格 [F, 24*24, 1536]
    current(末帧) -> fine pool(64);history(前 F-1 帧) -> coarse pool(4/帧)
    _pad_history_frames 把 coarse 补到 31 帧(复制最旧帧);构造 time_indices
    _embed_visual_bundle(复用 features-in):projector + 插 temporal marker + 末尾 control_query
      -> [221, hidden],vLLM 标准 scatter 填占位符
  backbone(MiniCPMModel) + LAST pooling(control_query)+ trajectory head -> [24]
```

**与 features-in 的关系**:pixels-in 只是把「客户端算 coarse/fine 特征」换成「tower 内算」,**池化后完全复用 `_embed_visual_bundle`**(projector / marker / control 逐项一致),下游 backbone/pooler/head 不变。

---

## 5. 关键实现点(file: 符号)

`vllm/model_executor/models/minicpm_robottrack.py`:

1. **`DualVisionTower(nn.Module)`**:无状态逐帧纯函数,两个编码器均为 **in-tree vLLM 模块**。
   - `dino = DINOv3VisionModel(dino_cfg, quant_config, prefix="dino")`(§5.2 的原生端口);`siglip = SiglipVisionModel(siglip_vision_cfg, require_post_norm=True, use_head=False, prefix="siglip")`(vLLM 原生 SigLIP;`require_post_norm=True` 复现 HF `last_hidden_state` 的 post-LN,`use_head=False` 丢注意力池化头)。
   - **外挂权重加载**:`_iter_safetensors(path)` 直接读本地 checkpoint 的 safetensors,喂给各自 `load_weights`;DINOv3 名称与 checkpoint 一致(直接命中),SigLIP 剥掉 `vision_model.` 前缀、跳过 `head.`/`text_model.`,由 vLLM `SiglipVisionTransformer.hf_to_vllm_mapper` 做 q/k/v→qkv_proj 融合。
   - `forward(dino_pixels, siglip_pixels) -> (fused[F,24*24,1536], grid)`:DINO 输出(post-LN 全 token)丢 `1+num_register`→24×24;SigLIP `adaptive_avg_pool2d` 到 24×24;`cat((dino, siglip), -1)`(**DINO 在前**)。
   - `dtype`/`device` 属性用 `next(self.parameters())`。

### 5.2 DINOv3 原生端口(`vllm/model_executor/models/dinov3.py`)

vLLM 无原生 DINOv3,故新写 in-tree 端口,忠实复刻 transformers `DINOv3ViTModel`(`dinov3_vit`):
- **结构**:patch-conv embeddings + CLS + register token;**2D 轴向 RoPE 只作用于 patch token**(CLS/register 跳过);LayerScale(`lambda1`)残差块;末尾 LayerNorm 出 `last_hidden_state`。
- **vLLM 原生层**:q/k/v/o、mlp up/down 用 `ColumnParallelLinear`/`RowParallelLinear`(TP/量化可组合);`k_proj` 无 bias(`key_bias=false`)。注意力数学用 `F.scaled_dot_product_attention`(双向、无 KV cache、自包含,不依赖 encoder-attn 元数据)。MLP 用 `nn.GELU()`(精确 erf,匹配 HF `ACT2FN["gelu"]`)。
- **RoPE**:`inv_freq = 1/base^arange(0,1,4/head_dim)`(base=100),patch 中心坐标归一化到 [-1,1],`angles = 2π·coords·inv_freq` → flatten → `tile(2)` → cos/sin;`apply_rotary_pos_emb` 对 patch 段做 `x·cos + rotate_half(x)·sin`。分辨率动态(按实际像素算 patch 网格),384 → 24×24。**对 HF RoPE 逐值 `0.0`**(见 test_dinov3)。
- **权重名**:`embeddings.{cls_token,register_tokens,patch_embeddings.*}`、`layer.{i}.{norm1,attention.{q,k,v,o}_proj,layer_scale1,norm2,mlp.{up,down}_proj,layer_scale2}`、`norm.*` — 与 checkpoint 一致;`load_weights` 用 `AutoWeightsLoader(skip_substrs=["mask_token"])`(mask_token 推理无关;inv_freq 计算得出、不在 checkpoint)。

2. **`_grid_pool` / `_pad_history_frames` / `_normalize_windows`**(模块级、可单测):
   - `_grid_pool([B,grid²,C], grid, out)` → `[B,out,C]` adaptive avg pool。
   - `_pad_history_frames([n,coarse_per,C], 31)`:复制最旧帧补到 31、取最近 31(= 上游 `data.py` / `assemble_window` 的补齐)。
   - `_normalize_windows(frames)`:把 processor 传来的 `frames` 归一成「每 item 一个窗口」的列表——**profiling 的 dummy 会比真实请求多套一层**(单窗口 vs 窗口列表),此处统一。

3. **`MiniCPMRobotTrackPixelItems(ModalityDataItems)`**:整窗口 = 1 个 item(`get_count()==1`);`get_processor_data() -> {"frames": self.data}`(把原始帧路由进 `_call_hf_processor`),`get_passthrough_data() -> {}`。

4. **`MiniCPMRobotTrackDataParser._parse_image_data`**:分派——`list/tuple` 或 `dict` 含 `"frames"` → pixels-in;`dict` 含 `coarse_tokens` → features-in(原样保留)。

5. **`MiniCPMRobotTrackProcessingInfo`**:
   - `get_dino_processor()` / `get_siglip_processor()`:惰性加载 `AutoImageProcessor` / `SiglipImageProcessor`(纯 config,无权重),缓存在 info 实例上。
   - `prepare_pixels(window)`:每帧 RGB + resize384 BICUBIC → 两个 processor 归一化 → `(dino_pixels, siglip_pixels)`。忠实复刻上游 `_prepare`。
   - `get_num_pixel_image_tokens() == 221`(tower 内部补到 31 帧,故占位符数固定,与客户端发多少帧无关)。

6. **`MiniCPMRobotTrackMultiModalProcessor`**:
   - `_call_hf_processor`:tokenize + (有 frames 时)逐窗口 `prepare_pixels`,输出 `dino_pixels/siglip_pixels/frame_lengths`;无 frames 时只 tokenize(features-in 原样)。
   - `_hf_processor_applies_updates -> False`(占位符一律由 `_get_prompt_updates` 插;对 pixels-in item 必须显式关掉,否则框架以为 HF processor 已插)。
   - `_get_mm_fields_config`:含 `dino_pixels` 时用 `flat_from_sizes("image", frame_lengths)`(dino/siglip)+ `batched` frame_lengths;否则回落 `_robottrack_field_config`(features-in)。
   - `_get_prompt_updates`:pixel item → 221;feature item → 原逐 run 统计。

7. **`MiniCPMRobotTrackModel`**:
   - `__init__`:tower 在 `with self._mark_tower_model(vllm_config, "image")` 内构造(标记为 tower 组件),随后 `self.vision_tower.to(device, dtype)`(engine 设备/精度)。
   - `_encode_window(dino_px, siglip_px)`:tower → current fine(64)/ history coarse(4/帧)→ `_pad_history_frames` 到 31 → 造 `coarse_time/fine_time` → 返回 `(coarse[124,1536], coarse_time, fine[64,1536], fine_time)`。
   - `_embed_pixel_windows`:按 `frame_lengths` 切窗口,逐窗口 `_encode_window` → `_embed_visual_bundle`。
   - `embed_multimodal`:**含 `dino_pixels` → pixels 分支;否则 features 分支**(同批次单模)。
   - `load_weights`:`AutoWeightsLoader(self, skip_prefixes=["vision_tower."])` + **把 `vision_tower.*` 参数名并入已加载集合**——tower 权重在 `__init__` 里从外挂 checkpoint 自加载(不在 RobotTrack checkpoint 里),否则 vLLM 的「缺失权重」校验会报错。

---

## 6. 踩坑(均已修)

1. **tower 权重外挂加载**:DINOv3/SigLIP 不在 RobotTrack checkpoint,原生模块在 `__init__` 里读外挂 safetensors 喂给各自 `load_weights`;顶层 `load_weights` 必须把 `vision_tower.*` 参数并入已加载集合,否则 `_check_weights` 报「缺失权重」。
2. **SigLIP `last_hidden_state` 复现**:vLLM `SiglipVisionModel` 默认可能省 post-LN、带池化头;须 `require_post_norm=True` + `use_head=False` 才等于 HF `.last_hidden_state`(post-LN 的 patch token)。加载时剥 `vision_model.` 前缀并跳过 `head.`/`text_model.`。
3. **DINOv3 RoPE 精度**:HF 在 fp32 下算 RoPE(`maybe_autocast enabled=False`)再 cast;端口同样先 fp32 再 cast,对 HF 逐值 `0.0`。`nn.GELU()`(精确 erf)对齐 HF `ACT2FN["gelu"]`,勿用 tanh 近似。
4. **profiling dummy 帧多套一层**:profiling 传来的 `frames` 是 `[window]`(1 元素、元素是 32 帧列表),真实请求是扁平 `window`。→ `_normalize_windows` 统一成「窗口列表」再逐窗口编码。
5. **vLLM 原生层需分布式上下文**:`ColumnParallelLinear` 等读全局 TP state;tower 在引擎内 `__init__`(dist 已初始化)构造,故 OK;纯脚本单测原生 tower 需自行 init 分布式(本仓用 in-engine e2e 对拍,RoPE 数学则纯 torch 单测)。

---

## 7. 验证

| 层级 | 方法 | 结果 |
|---|---|---|
| DINOv3 RoPE 逐值 parity | 端口 `DINOv3ViTRopePositionEmbedding`/`apply_rotary_pos_emb` vs HF `dinov3_vit`(纯 torch) | ✅ **`0.0`** |
| DINOv3 数学单测 | `test_dinov3.py`:rotate_half / rope 形状 / prefix 不变 / 旋转保范 | ✅ |
| 纯逻辑单测 | `_square_side`/`_grid_pool`(= 手工均值)/`_pad_history_frames`/窗口 221 不变量 | ✅ |
| 引擎起效 | pixels-in 真实图片跑通、存 BEV 叠加图(exit 0) | ✅ |
| **端到端 parity** | 同 8 帧:pixels-in(**vLLM 原生 tower**)vs features-in(上游 `DualVisionEncoder`),vLLM 24 维输出 | ✅ **max abs diff 4.2e-3** |
| 输出响应性 | 24 维范数随帧 0.14 → 0.63 递增,非恒零 | ✅ |
| Lint | `ruff check` / `ruff format` | ✅ |

> parity 脚本要点:同一个 `LLM` 实例同时支持两条路;客户端用上游忠实 `DualVisionEncoder` 出 coarse/fine 走 features-in 得 `traj_A`,发原始窗口走 pixels-in(原生 tower)得 `traj_B`,对拍 `|traj_A - traj_B|`。逐帧 ~0.3–4e-3 差来自两次独立 fp32 编码器前向的非确定性(SDPA vs eager、pool/矩阵乘顺序)。等价链:原生 DINOv3 ≈ HF DINOv3、vLLM SigLIP ≈ HF SigLIP,且 features-in 已对 HF golden 验证。

---

## 8. 运行方式

### 真·图片输入 + BEV 可视化(pixels-in)
```bash
CUDA_VISIBLE_DEVICES=6 python examples/pooling/robottrack_minicpm_video.py \
    --model  /cache/zhanghao/model/MiniCPM-RobotTrack/ \
    --dino   /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
    --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
    --images track-image/0 --output output --bev-range 1.0 \
    --instruction "Follow the person."
```

### 引擎入口(pixels-in)
```python
from vllm import LLM
import numpy as np

llm = LLM(model=".../MiniCPM-RobotTrack", runner="pooling", dtype="float32",
          enforce_eager=True, enable_mm_embeds=True, limit_mm_per_prompt={"image": 1},
          hf_overrides={"dino_model": ".../dinov3-vits16-pretrain-lvd1689m",
                        "siglip_model": ".../siglip-so400m-patch14-384",
                        "image_size": 384})
window = {"frames": [np.asarray(img) for img in rolling_deque]}  # <=32 原始帧
out = llm.embed([{"prompt": instr, "multi_modal_data": {"image": window}}])
traj = torch.tensor(out[0].outputs.embedding).reshape(8, 3)
```

> features-in(方案 A)入口不变:`mm_data={"image": {"coarse_tokens":..., "fine_tokens":..., ...}}`。

### 门控测试
```bash
# pixels-in e2e(需本地 checkpoint + 编码器权重)
MINICPM_ROBOTTRACK_PATH=.../MiniCPM-RobotTrack \
DINOV3_MODEL_PATH=.../dinov3-vits16-pretrain-lvd1689m \
SIGLIP_MODEL_PATH=.../siglip-so400m-patch14-384 \
.venv/bin/python -m pytest tests/models/multimodal/pooling/test_minicpm_robottrack.py -v
```

---

## 9. 已知限制 / 下一步

- **无 encoder cache**:每步重编码整窗口;要复用历史帧编码 → 升级 **B-2**(每帧一 item + marker-item,命中框架 `encoder_cache`,tower 仍无状态)。
- **同批次混用 pixels-in 与 features-in 不支持**:`embed_multimodal` 按是否含 `dino_pixels` 整批分派;单客户端单模无影响。
- **tower 注意力用 SDPA、默认 eager / fp32**:原生 linear 已支持 TP/量化,但注意力未走 vLLM `MMEncoderAttention`,也未接 encoder CUDA graph;大 TP / fp8 / cudagraph 优化待后续。DINOv3 仅在 `dinov3_vit`(RoPE 版)配置下验证,其他变体(gated MLP 等)未测。
- **编码器权重外挂**:DINOv3/SigLIP 不在 RobotTrack checkpoint,须 `hf_overrides` 指定路径;版本矩阵(vLLM+transformers+权重)更复杂。
- **PR / 人工问责**:按 `AGENTS.md`,AI 辅助工作需人工逐行 review、跑测试;PR 描述须含非重复性 / 测试结果 / 模型评测 / AI 辅助声明。

---

## 10. 版本 / 环境备注

- 主环境:本仓库 `.venv`,transformers **5.14.1**,torch **2.11.0+cu130**。
- tower 编码器**不再用 transformers 模型**(原生 vLLM);`transformers` 仅用于:`AutoConfig.from_pretrained` 读 encoder config、`AutoImageProcessor`/`SiglipImageProcessor` 做 CPU 归一化。
- DINOv3 checkpoint(`facebook/dinov3-vits16-pretrain-lvd1689m`,`model_type=dinov3_vit`,hidden 384 / 12 层 / 6 头 / patch16 / 4 register / `rope_theta=100` / `use_gated_mlp=false`);SigLIP(`google/siglip-so400m-patch14-384`,27×27=729 patch,hidden 1152)。均以本地 safetensors 直接加载权重。
