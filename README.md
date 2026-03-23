# 基于vLLM框架的AutoMTP简易实验环境文档

本文档说明仓库中 `base_vllm/`相对vLLM V1 在Speculative Decoding方向上的AutoMTP算法优化内容，便于对接的兄弟快速理解算法逻辑、数据流、配置项与debug设置。核心逻辑：通过在 **mtp module**中可选挂载 **Stop Head adapter（提前结束draft token proposing）** 以省算力（主要在于节省target model证伪tokens时所用的FLOPs）。

---

## 1. `EagleProposer` 中的关键行为
### 1.1 Stop Head 仅作用于读取speculative_config model参数路径去下的权重中有 adapter key的情况：
拿Qwen3-8B Eagle3举例：

`use_should_stop` 条件大致为：

- `method == "eagle3"` 且模型为 `Eagle3LlamaForCausalLM` 且 **`self.model.adapter is not None`**
- 且未设置 `VLLM_EAGLE_DISABLE_STOP=1`

### 1.2 提前结束草稿循环（Early exit）

每得到一个draft token的logits和hidden 后，可在分支CUDA stream上执行 should_stop（或与主串行，见环境变量）。若 `should_stop` 掩码 **`any()` 为真**，则 **break** 出循环，早停draft token propose过程。

- **统一形状**：目前vLLM对变长draft token proposing的兼容性（？），为了快速验证吞吐量优势，因此简化了早停思路。

### 1.3 Stop 路径里 CUDA Graph **capture 的意义**

should_stop 在算子上是「**小 subgraph**」：softmax、entropy、concat、小型 MLP adapter、sigmoid、比较等，**单次 FLOPs 远小于整段 draft forward**，但会拆成 **很多次 GPU kernel / 很多次 CPU 侧 launch**。在 **每一步 `propose`** 里都要跑一遍，所以我理解有不小的CPU的开销和图的开销。

**CUDA Graph capture 做的事**：在 **固定的拓扑与固定的张量形状**（由 `batch_size` 决定 `logits_buf [B,V]`、`hidden_buf [B,H]`）下，把这条算子链 **事先整段录制** 进一张 graph；运行时只做 **`copy_` 填输入 + `graph.replay()`**

- **降低 should_stop 的逐步尾延迟**，让这条短链在 **侧流上与主 stream 草稿前向重叠** 时更有实际收益；  

 

### 1.4 `should_stop` 核心逻辑（`eagle.py` / `EagleProposer.propose`）

**1）是否启用**

仅当 `use_should_stop` 为真时进入本节逻辑（条件同 §1.1：`eagle3` + `Eagle3LlamaForCausalLM` + `adapter is not None`，且未设 `VLLM_EAGLE_DISABLE_STOP=1`）。纯 MTP 草稿无 adapter，不走 should_stop。

**2）在 propose 里何时跑**

- 位于 `for token_index in range(num_speculative_tokens - 1):` 内，且 **仅在 `token_index == 0`** 时执行一次。  
- 由于advanced drafter通常在第一个token的接收率较高，因此仅在token2位置做gate（效率与性能较好的trade-off）。

**3）算子链（与 `llama_eagle3` adapter 一致）**

```
logits [B, V]
    -> softmax -> 按词表维求 entropy（标量/行）-> [B, 1]
    -> concat(hidden_states [B, H], entropy) -> [B, H+1]
    -> adapter (Stop Head) -> stop_logits
    -> sigmoid -> >= 0.5  -> should_stop_mask [B] (bool)
```
是否走 Graph 由 `VLLM_EAGLE_STOP_CUDAGRAPH` 与是否已有对应 `batch_size` 的缓存条目共同决定（见下）。

**4）并行路径（默认 `VLLM_EAGLE_STOP_SERIAL=0`）**

目标：在 **刚算完第 2 个 draft token** 之后，把 should_stop 放在 **`_stop_stream`** 上与主 stream 上的 **后续 draft 前向（第 3、4…个 token）** 尽量重叠（若未 early exit）。

- **循环结束后的收尾**  
  - `propose` 在 **`torch.stack(draft_token_ids_list)`** 之前若 **`_stop_done_event` 仍非 `None`**，主 stream 会再 **`wait_event(_stop_done_event)`** 一次（主要是防止cornercase路径下 should stop 未在循环内耗尽）。

**5）串行对比路径（`VLLM_EAGLE_STOP_SERIAL=1`）**

- 仍在 **`_stop_stream`** 上做 `replay()` 或 eager（与并行相同的 graph/buffer 逻辑），但在 launch 后立刻 **`_stop_stream.synchronize()`**，主线程阻塞到 should_stop 算完。  
- **`_stop_done_event` 置为 `None`**，后续不再用 event 与主 stream 异步握手。  
- 用途：**消除与「第 3、4 个 draft」主 stream 前向的时间重叠**，便于 A/B 看 stop 本身耗时或验证正确性。

**6）CUDA Graph 捕获与运行**

**数据结构**

- **`self._stop_graphs: dict[int, tuple]`**，键为 **`batch_size`**。  
- 每个值为 **`(graph, logits_buf, hidden_buf, mask_buf)`**：  
  - `logits_buf`：`[B, V]`，`V = model_config.get_vocab_size()`  
  - `hidden_buf`：`[B, H]`，`H` 为草稿 hidden 维  
  - `mask_buf`：Graph 在 capture 中执行 **`stop_prob >= 0.5`** 的输出，**replay 时原地写入同一 buffer**，主流程里 **`_should_stop_mask` 即指向 `mask_buf`**（通过 replay 更新内容）。

**`_capture_stop_graph(batch_size)`**

- 若该 `batch_size` 已存在或当前不是 `eagle3` + 有 `adapter`，直接返回。  
- 在 **`torch.inference_mode()`** 下捕获（避免 adapter 与 inference tensor 的 inplace 报错）。  
- **Warmup**：先在 `_stop_stream` 上对全零的 `logits_buf`/`hidden_buf` 跑一遍与 capture 相同的算子链，再 **`synchronize()`**（CUDA Graph 要求）。  
- **Capture**：`with torch.cuda.graph(graph, stream=_stop_stream)` 内再跑同一算子链，得到 **`mask_buf`**；**不**把 cross-stream 的 wait/record 录进 graph。  
- 最后 **`self._stop_graphs[batch_size] = (graph, logits_buf, hidden_buf, mask_buf)`** 并打 log。

**`pre_capture_stop_graphs()`（worker warmup）**

- 若 **`VLLM_EAGLE_STOP_CUDAGRAPH != "1"`** 或无 adapter，直接返回。  
- 默认对一组 **`batch_size`**：`1, 2, 4, 8, 16, 32, 64, …, 256`（完整列表见源码），再 **`<= scheduler_config.max_num_seqs`** 过滤，逐个调用 **`_capture_stop_graph(bs)`**。  
- 可用 **`VLLM_EAGLE_STOP_PRECAPTURE_SIZES="1,2,8,..."`** 覆盖默认列表。  
- 设计意图：**首请求前** 就具备常见 B 的 graph，避免第一次命中该 batch 时的捕获延迟。

**运行时选路（`propose` 内，`token_index==0`）**

1. **`stop_entry = _stop_graphs.get(batch_size)`** 命中：  
   - **`copy_` 输入 →（并行）main `record_event` + stop `wait_event` + `replay()` + `done_event.record()`**；或（串行）`replay()` 后 **`_stop_stream.synchronize()`**。  
2. **未命中** 且 **`VLLM_EAGLE_STOP_LAZY_CAPTURE=1`**：当场 **`_capture_stop_graph(batch_size)`**，再按上条 replay；并置 **`_did_lazy_capture`**，**`VLLM_EAGLE_STOP_PROFILE` 统计会跳过本轮**（避免把捕获时间算进 propose）。  
3. **仍未命中** 或 **`VLLM_EAGLE_STOP_CUDAGRAPH=0`**：**eager**：并行路径下对 **`logits.clone()` / `hidden_states.clone()`** 在侧流上做 softmax/entropy/adapter（同样先 `wait_event` 再算，并可选 **`_stop_done_event`**）。

**`VLLM_EAGLE_STOP_LAZY_CAPTURE=0`（默认）时**：仅对 **预捕获列表里有的 `batch_size`** 使用 Graph；其它 B **一直 eager**，避免线上突增延迟；需要全覆盖时再打开 lazy 或扩大 `PRECAPTURE_SIZES`。


**7）流程缩略**

```
[循环内: 已完成第 2 个 draft 的 forward+logits+append]
                    |
                    v
         use_should_stop 且 token_index==0 ?
                    |
        否 ---------+------------------ 是
        |                                |
        v                                v
   （不跑 stop）              串行或并行跑: graph / lazy capture / eager
        |                                |
        +----------------+----------------+
                         v
              mask.any() ? -- 是 --> break（不再生成更多 draft）
                         |
                         否 --> 继续下一 token_index
                         |
                         v
         （循环外）若有 _stop_done_event：主 stream wait_event
```

### 1.5 Propose时间统计（通过环境变量控制是否需要）

- **`VLLM_EAGLE_STOP_PROFILE=1`**：用 CUDA Event 统计 `propose` _wall、draft 前向、should_stop 占比等，每隔 `VLLM_EAGLE_STOP_PROFILE_INTERVAL`（默认 20）打日志。 
---

## 2. Stop Head 权重与 `llama_eagle3.py`

- Checkpoint 中若 **没有** `adapter` 相关权重：`load_weights` 将 **`self.adapter = None`**，此时等价于 **关闭 Stop Head**（`should_stop` 全 False）。
- 若有 adapter 权重：实例化 **`AutoMTPStopHeadMid`**（hidden + 1 维熵输入，与 `Qwen3RMSNorm` 等一致）。
- **`VLLM_EAGLE_DISABLE_STOP=1`**：强制不启用 stop 逻辑，便于快速 A/B test。

---

## 3. 环境变量速查

| 变量 | 默认 | 含义 |
|------|------|------|
| `VLLM_EAGLE_STOP_SERIAL` | `0` | `1` 时 should_stop 走主 stream 串行，便于对比重叠效果 |
| `VLLM_EAGLE_STOP_CUDAGRAPH` | `1` | `0` 关闭 stop 子图 |
| `VLLM_EAGLE_STOP_LAZY_CAPTURE` | `0` | `1` 允许在 propose 时对未见 batch size 懒捕获 |
| `VLLM_EAGLE_DISABLE_STOP` | `0` | `1` 禁用 Stop Head |
| `VLLM_SD_TIMING` | `0` | `1` 启用 runner 级 spec decode 分段计时 |
| `VLLM_SD_TIMING_INTERVAL` | `20` | 计时日志间隔 |

---

## 4. 与 `scripts/test_eagle3.py` 的对应关系

### 4.1 无SD基线

```bash
python scripts/test_eagle3.py --mtp_size 0 --batch_size 1 --data deepmath
```

`mtp_size == 0` 时不传 `speculative_config`，为普通自回归。

### 4.2 MTP 投机（主模型自带 MTP）

```bash
python scripts/test_eagle3.py --mtp_size 4 --batch_size 1 --data deepmath
```

### 4.3 AutoMTP

```bash
python scripts/test_eagle3.py --stop --mtp_size 4 --batch_size 64 --data deepmath --stop
```

### 4.4 批处理与吞吐指标

- **`--batch_mode batch_requests`（默认）**：同一 prompt 复制 `batch_size` 份，独立请求，测 **批量吞吐**。
- 脚本会调用 **`llm.get_first_completion_throughput()`**（本 fork 在 `LoggerManager` 中实现）并写入 `scripts/generation_log/` 下汇总 txt。
