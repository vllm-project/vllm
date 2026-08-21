# vLLM Weight Reload Refactoring RFC (Review Draft)


## 1. Motivation

A checkpoint tensor and its serving tensor may differ in dtype, shape, stride, layout, and even parameter count:

```text
checkpoint -> name mapping -> TP/EP sharding -> fusion/padding
           -> transpose/layout conversion -> quantization/repack
           -> PWAL/derived state -> serving tensor
```

A reload therefore cannot copy every received tensor directly into the current runtime parameter by name. Runtime parameters may already be fused, packed for Marlin, transformed by PWAL, or accompanied by MLA-derived tensors. At the same time, captured CUDA Graphs require stable parameter objects and storage addresses.

## 2. Refactoring principles and lifecycle

This design does not use layerwise reload and does not restore the entire model to a meta/checkpoint schema for rematerialization. Reload always runs after requests have drained: `drain -> update -> resume`. Hot-swapping weights while requests are executing is out of scope. `START` and explicit `FINISH` define an update; layer boundaries, transport chunks, and `numel` do not.

The design has four invariants:

1. `data_ptr()` remains stable and CUDA Graphs are not recaptured. Writes use `copy_`; shape-changing paths use `resize_ + copy_`.
2. Reload always calls the model's native `model.load_weights()` and reuses AutoWeightsLoader, model-specific name routing, TP/EP sharding, and stacked/expert mappings.
3. Reload does not use layerwise meta restoration, rematerialization, kernel tensors, or a generic online-process loader.
4. A dtype mismatch without an explicit online quantization loader fails closed. An incomplete manifest also fails closed.

```text
START
  drain requests
  call restore_weights_before_loading for shape-changing quant methods
  resize parameters to checkpoint shapes and reset conversion guards
DATA*
  receive checkpoint-format tensors
  call model.load_weights([(name, tensor)])
  let param.weight_loader perform copy/transpose/repack/online quantization
FINISH
  run selective PWAL for restored shape-changing layers
  run refresh_derived_state on applicable modules
  run manifest_check(expected, loaded)
  resume requests
```

The design introduces two extension points. Extension A is the permanent per-parameter `weight_loader`, which performs eager parameter processing. Extension B is per-module `refresh_derived_state()`, which rebuilds derived values at finish. The transport layer supplies checkpoint-format tensors and does not implement model transformations.

## 3. Scope overview: two reload paths

Weight reload is divided into two connected but clearly bounded scopes. **V1** covers cases where the sender already provides the dtype and checkpoint format required by serving. Its primary concern is to reuse the native loader, PWAL, and derived-state logic while keeping runtime storage stable. **V2** covers cases where trainer and serving dtypes differ. It keeps the same reload lifecycle but adds online quantization loaders so conversion happens while weights are received, rather than after an entire layer or model has been staged.

In short, V1 answers how to safely update format-matched weights, while V2 answers how to update weights that require online conversion. Pre-quantized checkpoints remain in V1. BF16 trainer to FP8/FP4/INT8 serving belongs to V2 and requires an explicit online quantization loader. The two scopes are detailed below.

## 4. V1 scope: matching dtype and checkpoint format

V1 assumes the sender already provides the dtype and checkpoint format required by the serving configuration. It reuses existing loaders, PWAL, and derived-state logic, but does not implicitly convert BF16 weights into a low-precision serving format.

| Scenario | Scope | Start | Receive (`weight_loader`) | Finish |
|---|---|---|---|---|
| BF16 dense/MoE with a BF16 trainer | V1 | None | `copy_` | None |
| Block FP8 with an FP8 trainer | V1 | None | `copy_` | None |
| Per-tensor FP8 dense with an FP8 trainer | V1 | None | transpose + `copy_` | None |
| Per-tensor FP8 MoE with an FP8 trainer | V1 + PWAL | Restore checkpoint shape | Copy checkpoint-format values | Scale requantization, kernel shuffle, and `_g1_alphas` refresh |
| MLA models such as DeepSeek V2/V3/R1 and Kimi K3 | V1 + derived state | None | Copy `kv_b_proj` | Refresh `W_UK_T/W_UV`; MXFP4/FP8 backends use their corresponding re-quantization path |
| Pre-quantized INT4/Marlin checkpoint | V1 + PWAL | Restore checkpoint shape | Copy GPTQ/AWQ-format values | Marlin repack |
| Block FP8 serving with a BF16 trainer | V2 | None | Online loader performs per-block quantization + `copy_` | None |
| Per-tensor FP8 serving with a BF16 trainer | V2 | None | Online loader plus fused-parameter unit buffer | Finalize only unified-scale units not completed eagerly |
| NVFP4 serving with a BF16 trainer | V2 | Backend dependent | Online loader plus the smallest required closure buffer | Layout shuffle or derived-state refresh when required |
| BF16 trainer with low-precision serving but no online loader | Reject | None | Raise `ValueError` | None |

The main V1 prerequisite is replacing shape-changing `replace_parameter` calls in PWAL with `resize_ + copy_`. Quantization methods record original shapes during `create_weights`, implement `restore_weights_before_loading()`, and use an idempotent conversion guard. Non-shape-changing PWAL does not need to run again. Derived-only work moves into `refresh_derived_state()`.

## 5. V2 scope: online quantization

V2 preserves the V1 lifecycle and permanently attaches online quantization to parameter loaders during `create_weights`. When `model.load_weights()` receives a BF16 or FP16 checkpoint tensor, the loader converts it into the serving format and writes it into stable runtime storage. Cold loading and reload therefore share the same conversion implementation.

| Input -> serving | V2 path | Buffer requirement |
|---|---|---|
| BF16 -> block FP8 | Per-block cast + `copy_` | No layer-level buffer; each shard is independent |
| BF16 -> per-tensor FP8 | Retain the shards sharing a scale, then quantize with the unified scale | Fused parameters require a unit buffer |
| BF16 -> NVFP4 | Collect the smallest required closure, such as w1+w3, then quantize and convert layout | One buffer per incomplete unit |
| BF16 -> MXFP8/INT8 | Quantization-method-specific online loader | Determined by scale granularity |

A serving quantization configuration without a matching online loader is rejected. Pre-quantized checkpoints such as GPTQ and compressed-tensors remain V1 checkpoint-format/PWAL paths.

## 6. Shared V1/V2 responsibilities

* **Start:** drain requests and restore only shape-changing layers, not the whole model schema.
* **Manifest:** declare tensor names, checkpoint dtypes and shapes, logical shards, quantization format/version, and expected coverage.
* **Loader:** perform model-specific mapping and sharding; a V2 loader additionally performs online quantization.
* **Selective PWAL:** process only restored layers requiring shape or layout conversion. A transport chunk must never trigger whole-layer PWAL prematurely.
* **Derived state:** refresh MLA and similar values after all dependencies arrive.
* **Finish:** complete selective PWAL and derived-state refresh before manifest validation. Serving does not resume after any failure.
* **Storage:** perform conversions in the original parameter storage, preserving Python identity, `data_ptr`, and CUDA Graph validity.

## 7. Smallest quantizable unit

Automatic detection operates on a `QuantizationUnit`, not on an arbitrary parameter or network chunk. A unit is the smallest input set satisfying all of these conditions:

1. All required logical shards have arrived.
2. The complete scale domain is available. A per-block domain is local, while a per-tensor domain may span shards.
3. Quantization, transpose/layout conversion, and fusion depend only on this set.
4. The result and its scale/metadata can be written into the target serving slice as one operation.

The quantization method and parameter loader jointly discover units. The loader derives logical keys from expert ids, TP shards, stacked parameters, and scale parameters. A tracker determines completion with key-set coverage, never tensor counts or `numel`.

In the reference implementation, a quantization method exposes `reload_units(layer)` to declare unit keys, staged parameters, and a finalize callback. A shard-aware wrapper around the normal loader maps global expert ids to local expert ids and reports `(parameter, local expert, shard id)` keys. Quantization semantics remain owned by the quantization method, while actual shard arrivals remain owned by the model loader. The framework does not hard-code Q/K/V, w1/w2/w3, or individual quantization formats.

## 8. Eager processing and buffer lifetime

```text
receive one shard
  -> validate manifest/key/checksum
  -> does this shard contain a complete quantization domain?
       yes: quantize + copy_, then release the received tensor
       no: let the quant tracker retain the shard and update coverage
           -> is the unit complete?
                no: wait for the remaining shards
                yes: quantize/requantize + convert layout
                     write the serving slice and scale
                     release retained shards after the CUDA event completes
```

Only units that require multiple shards need staging. The relevant quantization tracker creates and owns this storage lazily; the protocol is not tied to `reload_arena` or any global allocator. Per-block, per-group, per-channel, and non-fused per-tensor paths consume the current shard directly. Fused per-tensor paths retain only shards that are still waiting for their peers.

The per-expert reference implementation lazily creates one checkpoint-format slab for each staged `(parameter, local expert)` slot. The quantization method explicitly declares its shape and dtype because they cannot be inferred from the runtime parameter after PWAL. For example, a checkpoint scale can have shape `[E, 2]` while its runtime form has shape `[E]`. The tracker exposes the slab through an expert-dimension broadcast proxy so the original loader can continue to perform TP narrowing, expert indexing, and fusion. Finalizing the unit removes both slab and proxy. The allocator remains an implementation detail.

Asynchronous quantization must protect input lifetime with a CUDA event. Retained inputs are released as soon as the kernel no longer reads them. Peak additional memory is proportional to unmatched shards, not the checkpoint size of the complete layer or model.

## 9. Unit granularity matrix

| Granularity/parameter | Smallest processing unit | Cross-shard buffer | Processing point |
|---|---|---|---|
| Per-block: Block FP8, MXFP4, MXFP8 | One logical shard, processing its internal blocks independently | No | Quantize when the shard arrives |
| Per-group: GPTQ/AWQ group | A logical shard containing complete groups | No | Convert or quantize when the shard arrives |
| Per-channel: INT8/SmoothQuant | A logical shard containing the complete row/column scale domain | No | Quantize when the shard arrives |
| Per-tensor FP8, non-fused parameter | Complete local tensor shard | No | Compute the local tensor scale and quantize on arrival |
| Per-tensor FP8 fused QKV | Q/K/V shards sharing one serving scale | Yes | Quantize after all paired shards arrive |
| Per-tensor FP8 fused MoE `w13` | One expert's w1, w3, and corresponding scales | Yes | Requantize with a unified scale after both halves arrive |
| MoE w2 | One expert's w2 and scale | No | Process the serving slice on arrival |
| Layer-wide activation scale | All expert/layer contributions covered by the scale | Yes | After the dependency set completes, usually at `FINISH` |
| MLA-derived `W_UK_T/W_UV` | One MLA layer with all base tensors updated | Not a transport buffer | `refresh_derived_state()` |

If a quantization backend has scale or layout dependencies across an entire layer, it must not be forced into per-expert units merely to reduce memory. It remains deferred until `FINISH`.

### Backend applicability

A quantization method declares fine-grained units only when a serving value can be determined independently from one unit. For per-tensor FP8 MoE, Triton and batched-Triton on non-FNUZ platforms can finalize `w13` per expert. FlashInfer, Marlin, AITER, and DeepGEMM paths that require layer-wide shuffle, padding, or backend-owned conversion must not declare the same granularity and instead fall back to selective PWAL at `FINISH`. Eager-unit support is a quantization-method and backend capability, not something inferred from dtype alone.

The reference per-expert FP8 units are:

| Unit | Required keys | Staged values | Finalize |
|---|---|---|---|
| Per-tensor `w13[e]` | Expert e's w1/w3 weights and scales | `[2I, H]` weight and `[2]` scale | Requantize both halves with `max(s1, s3)` and write `w13[e]` |
| Block `w13[e]` | Expert e's w1/w3 weights and block scales | None | Checkpoint format is already serving format; the loader writes directly |
| `w2[e]` | Expert e's w2 weight and scale | None | The non-fused parameter is written directly |
| Static `input_scale` | Input scales from all experts | One scalar per expert | Take the layer-wide maximum at `FINISH` |

Unit finalization must share its quantization function with cold loading so streamed reload and cold-load conversion remain bit-identical.

## 10. Sharded transport

Because one transfer operation carries one sharding, each transport record needs a complete logical identity rather than only a parameter name:

```text
Record {
  update_id, sequence_no,
  tensor_name, logical_unit_id,
  tp/ep/expert/shard coordinates,
  byte_range, shape, stride, dtype,
  quant_format/version, checksum
}
```

The sender may pack records into buckets constrained by `max_chunk_bytes` and memory budgets. A bucket may cross parameters, experts, and layers. The receiver routes every record to its logical unit and never treats a bucket as a completion boundary. Out-of-order and duplicate records are deduplicated. At `FINISH`, missing shards are reported by unit and key.

The design adopts the memory-reduction objective discussed in SGLang issue 32335 but integrates it with vLLM's native reload path. A chunk is a communication and scheduling object; a quantization unit is the processing object:

```text
transport bucket -> model.load_weights(record)
                 -> unit tracker
                 -> unit complete: quantize + write + release retained inputs
                 -> FINISH: verify all units + deferred PWAL/derived work
```

The receiver applies backpressure when bytes retained by quantization trackers reach the configured budget. It pauses sending or reduces bucket size and prioritizes shards that complete nearly covered units. NCCL, CUDA IPC, and filesystem transfer share the same record schema. `(update_id, sequence_no)` identifies retransmissions.

## 11. Consistency and failure semantics

Requests have drained before reload begins, so eager writes cannot be observed by an executing forward pass. Immediate quantization and input release are memory optimizations inside the update window, not hot-swap or per-unit publication. Requests resume only after `FINISH` completes coverage validation, selective PWAL, derived-state refresh, and manifest validation.

An in-place update cannot cheaply roll back units already written. If `FINISH` detects a missing shard, conversion error, or OOM, the worker must not resume serving. The control plane must complete the same update or rebuild/reload the worker. Continuing to serve the old model after a failed update would require a complete shadow copy or double buffer and is outside this RFC's V1/V2 scope.

## 12. Verification requirements

Every enabled quantization backend must test out-of-order shards, units spanning buckets, duplicates and retransmission, missing keys, dtype mismatch, release of unit buffers, cold-load versus reload tensor/output equivalence, stable parameter identity, and stable `data_ptr`. A row in the V1/V2 matrix is considered supported only after its corresponding tests pass.

The reference implementation was validated on an H200 with a two-layer, 128-expert Qwen3-30B-A3B configuration converted to per-tensor FP8. It discovered 256 units per layer. Reloaded serving tensors were bit-identical to a cold load of checkpoint B by per-parameter SHA256; Parameter identity and storage-address changes were zero; generated tokens matched cold-served B and differed from A. Peak extra reload memory was 1256 MiB with streaming and 2409 MiB with streaming disabled.

The same checkpoint pair was also tested through vLLM's NCCL weight-transfer engine. A `vllm-rl-day0-kit` publisher on a second GPU sent 1567 tensors in five packed buckets. An expert's shards could cross bucket boundaries, yet key-set coverage completed 512 units across two layers and the updated output matched cold-served B. This validates that a transport bucket is not a completion boundary and that releasing expert staging after unit completion reduces peak memory.

---

# vLLM Weight Reload 重构 RFC（Review Draft）


本文档只讨论 weight reload 的重构方案。第一章以 `Weight Update Design v7` 为主体，定义整体流程、v1/v2 scope 和场景覆盖矩阵；第二章定义在线量化的最小处理单元，以及它与单 sharding 分片传输的关系。`docs/design/streaming_expert_reload.md` 仅作为细粒度单元跟踪的实现参考，不改变 v7 的 drain-update-resume 主流程。

## 1. Weight reload 整体重构

### 1.1 为什么必须重构

checkpoint tensor 与 serving tensor 可能在 dtype、shape、stride、layout 和参数数量上都不同：

```text
checkpoint -> name mapping -> TP/EP sharding -> fusion/padding
           -> transpose/layout conversion -> quantization/repack
           -> PWAL/derived state -> serving tensor
```

因此 reload 不能把接收的 tensor 直接按名称 `copy_` 到当前 runtime parameter。当前 runtime 还可能包含 PWAL 替换出的 tensor、fused 权重、Marlin packed 权重和 MLA 派生权重；同时 CUDA Graph 要求 parameter 对象及其 storage address 稳定。

### 1.2 重构原则与整体流程

本方案不采用 layerwise reload，也不把整个模型切换回 meta/checkpoint schema 后重新 materialize。reload 始终在请求排空后执行：`drain -> update -> resume`，不支持请求运行期间的 hot-swap。一次 update 的边界由 `START` 和显式 `FINISH` 定义，不由 layer、传输 chunk 或 `numel` 推断。

四项约束如下：

1. 全程保持 `data_ptr()` 不变，不重新 capture CUDA Graph；写入使用 `copy_`，shape-changing 路径使用 `resize_ + copy_`。
2. 始终调用模型原生 `model.load_weights()`，复用 AutoWeightsLoader、模型名称路由、TP/EP shard 和 stacked/expert mapping。
3. 不采用 layerwise 的 meta restore、materialize、kernel tensors 或通用 online-process loader。
4. dtype 不匹配且没有显式 online quant loader 时 fail-closed；manifest 不完整时 fail-closed。

整体调用流程为：

```text
START
  drain requests
  对 shape-changing quant method 调 restore_weights_before_loading
  resize_ 回 checkpoint shape，并重置 converted guard
DATA*
  接收 checkpoint-format tensor
  model.load_weights([(name, tensor)])
  param.weight_loader 执行 copy/transpose/repack/online quant
FINISH
  对已 restore 的 shape-changing layer 执行 selective PWAL
  遍历 module 执行 refresh_derived_state
  manifest_check(expected, loaded)
  resume requests
```

本方案只有两个扩展点：扩展点 A 是永久绑定在 parameter 上的 `weight_loader`，负责 per-parameter eager 处理；扩展点 B 是 module 的 `refresh_derived_state()`，负责 finish 阶段的派生状态刷新。传输层只提供 checkpoint-format tensor。

## 1.3 Scope 概览：两条 reload 路径

Weight reload 分为两个相互衔接但边界清晰的 scope。**V1** 面向发送端已经提供 serving 所需 dtype 和 checkpoint format 的场景，重点是复用原生 loader、PWAL 和派生状态逻辑，同时保证 runtime storage 稳定。**V2** 面向 trainer dtype 与 serving dtype 不一致的场景，在相同 reload 流程中增加 online quantization loader；量化发生在权重接收过程中，而不是等整层或整模型接收完成后再统一处理。

因此，V1 解决“匹配格式权重如何安全更新”，V2 解决“输入权重需要在线转换时如何更新”。预量化 checkpoint 仍属于 V1；BF16 trainer 到 FP8/FP4/INT8 serving 则属于 V2，且必须存在明确的在线量化 loader。下面分别说明两个 scope 的覆盖矩阵和处理路径。

### 1.4 V1 scope：dtype/格式匹配

v1 的原则是“发送端已经提供 serving 所需 dtype/格式”，reload 只复用冷启动已有的 loader、PWAL 和 derived-state 逻辑，不新增 BF16 到低精度的隐式转换。

| 场景 | Scope | Start | Receive (`weight_loader`) | Finish |
|---|---|---|---|---|
| BF16 dense/MoE + BF16 trainer | v1 | 无 | `copy_` | 无 |
| Block FP8 + FP8 trainer | v1 | 无 | `copy_` | 无 |
| Per-tensor FP8 dense + FP8 trainer | v1 | 无 | transpose + `copy_` | 无 |
| Per-tensor FP8 MoE + FP8 trainer | v1 + PWAL | restore checkpoint shape | 按 checkpoint format `copy_` | scale requantize + kernel shuffle + `_g1_alphas` refresh |
| MLA（DeepSeek V2/V3/R1、Kimi K3） | v1 + derive | 无 | `copy_` `kv_b_proj` | refresh `W_UK_T/W_UV`；MXFP4/FP8 backend 走对应重量化派生路径 |
| INT4/Marlin 预量化 checkpoint | v1 + PWAL | restore checkpoint shape | 按 GPTQ/AWQ format `copy_` | Marlin repack |
| Block FP8 serving + BF16 trainer | v2 | 无 | online loader：per-block quantize + `copy_` | 无 |
| Per-tensor FP8 serving + BF16 trainer | v2 | 无 | online loader + fused-param unit buffer | 只处理尚未 eager 完成的 unified-scale unit |
| NVFP4 serving + BF16 trainer | v2 | 视 backend layout 而定 | online loader + 最小闭包 buffer | layout shuffle/derived refresh（如需要） |
| BF16 trainer + 低精度 serving，但未配置 online loader | REJECT | 无 | `ValueError` | 无 |

v1 的关键改造是将 shape-changing PWAL 中的 `replace_parameter` 改为 `resize_ + copy_`，在 `create_weights` 时记录 original shapes，提供 `restore_weights_before_loading()`，并用 converted guard 保证 PWAL 幂等。非 shape-changing PWAL 不需要在 finish 重跑；derived-only 逻辑移入 `refresh_derived_state()`。

### 1.5 V2 scope：在线量化

v2 保留 v1 的 transaction 和调用栈，在 `create_weights` 阶段把在线量化逻辑绑定到 parameter 的 loader。于是 `model.load_weights` 收到 BF16/FP16 checkpoint tensor 时，loader 按量化方法把它转换为 serving tensor，再写入稳定 runtime storage；冷启动与 reload 共用同一转换实现。

| 输入 -> serving | v2 路径 | buffer 要求 |
|---|---|---|---|
| BF16 -> block FP8 | per-block cast + `copy_` | 每个 block 独立，零 layer buffer |
| BF16 -> per-tensor FP8 | loader 暂存共享 scale 所需 shard，统一 scale 后 requantize | fused 参数需要 unit buffer |
| BF16 -> NVFP4 | loader 收集量化所需闭包（例如 w1+w3）后转换/layout shuffle | 按最小 unit buffer |
| BF16 -> MXFP8/INT8 | 对应 quant method 的 online loader | 由 scale 粒度决定 |

v2 不支持“声明了 serving quantization 但没有对应在线 loader”的组合；该组合必须报错，而不是执行精度不正确的 `copy_`。预量化 checkpoint（GPTQ、compressed-tensors 等）仍走 v1 的 checkpoint-format/PWAL 路径。

### 1.6 V1/V2 共同流程与职责

* **Start**：drain 请求，只 restore shape-changing layer，不恢复整个模型 schema。
* **Manifest**：声明 tensor name、checkpoint dtype/shape、logical shard、量化格式/版本和 expected coverage。
* **Loader**：解释模型参数映射和 sharding；v2 loader 额外执行在线量化。
* **Selective PWAL**：只处理已 restore 且需要 layout/shape conversion 的 layer；不得因收到一个传输 chunk 就调用整个 layer 的 PWAL。
* **Derived state**：MLA 等派生值在所有依赖满足后 refresh。
* **Finish**：先完成 selective PWAL 和 derived refresh，再执行 manifest check；任何失败都不 resume serving。
* **Storage**：所有转换在原 parameter storage 上完成，保持 Python identity、`data_ptr` 和 CUDA Graph 有效性。

## 2. 细化设计：最小量化单元与分片传输

### 2.1 最小可量化单元

自动识别的对象不是“一个参数”或“一个网络 chunk”，而是 `QuantizationUnit`：满足以下条件的最小输入集合：

1. 所有需要的 logical sharding 已到齐；
2. scale 的统计范围已完整（per-block 可局部完成，per-tensor 可能跨 shard）；
3. 量化 kernel、transpose/layout 和 fused 规则只依赖该集合；
4. 可原子地写入目标 serving slice，并更新相关 scale/metadata。

Unit discovery 由 quant method 和 parameter loader 共同提供。loader 从自身的 expert id、TP shard、stacked parameter 和 scale 参数推导 logical key；tracker 以 key set coverage 去重和判定完成，绝不使用 tensor 数量或 `numel`。

参考实现中，quant method 通过 `reload_units(layer)` 声明每个 unit 的 keys、需要 staging 的 parameter 及 finalize callback；常规 loader 由 shard-aware wrapper 包装。wrapper 先把 global expert id 映射为 local expert id，再把 `(parameter, local expert, shard id)` 交给 tracker。这样 unit 的语义来自 quant method，shard 的实际到达信息来自模型 loader，框架不需要硬编码 q/k/v、w1/w2/w3 或具体量化格式。

### 2.2 单元完成后的即时处理

```text
收到一个 sharding
  -> 校验 manifest/key/checksum
  -> 当前 shard 是否构成完整量化域？
       是：直接 quantize + copy_，随后释放接收 tensor
       否：由 quant tracker 暂存该 shard 并更新 coverage
           -> unit complete?
                否：等待剩余 sharding
                是：quantize/requantize + layout conversion
                    写入 serving slice 和 scale
                    CUDA event 完成后释放该 unit 暂存的 shards
```

只有跨 shard 才能完成的 unit 才需要 staging buffer。该 buffer 由对应 quant method 的 unit tracker 按需创建和持有，不绑定 `reload_arena` 或其它全局 allocator：per-block、per-group、per-channel 以及 non-fused per-tensor 路径直接消费当前接收的 shard，量化完成后即可释放传输 tensor；fused per-tensor 路径只暂存尚未配对的 shard。

当前 per-expert 参考实现为每个需要暂存的 `(parameter, local expert)` 懒创建 checkpoint-format slab。slab 的 shape/dtype 必须由 quant method 显式声明，不能从 runtime parameter 反推，因为 PWAL 可能已经改变 shape（例如磁盘 scale 为 `[E, 2]`，runtime scale 为 `[E]`）。tracker 将 slab 暴露为带 expert 维度的 broadcast proxy，让原 `weight_loader` 继续执行 TP narrowing、expert indexing 和 fusion；unit finalize 后立即删除 slab 和 proxy。具体 allocator 是实现细节，未来可以替换，但不能成为 unit 协议的一部分。

异步量化时，tracker 必须用 CUDA event 保护输入生命周期，待 kernel 不再读取输入后立即释放。峰值额外显存取决于未完成配对的 shards，而不是整个 layer/model 的 checkpoint 大小。

### 2.3 典型 unit 规则

| 量化粒度/参数 | 最小处理单元 | 是否需要跨 shard buffer | 何时处理 |
|---|---|---|
| Per-block（Block FP8、MXFP4、MXFP8） | 一个 logical shard；其内部独立处理各 block | 否 | shard 到达后立即量化并释放输入 |
| Per-group（GPTQ/AWQ group） | 包含完整 group 的 logical shard | 否 | shard 到达后立即格式转换/量化 |
| Per-channel（INT8/SmoothQuant） | 包含完整行/列统计域的 logical shard | 否 | shard 到达后立即量化 |
| Per-tensor FP8 non-fused parameter | 完整 local tensor shard | 否 | shard 到达后计算 local tensor scale 并量化 |
| Per-tensor FP8 fused `qkv` | 共享 serving scale 的 q/k/v shards | 是 | 配对 shards 到齐后统一 scale 并量化 |
| Per-tensor FP8 fused MoE `w13` | 一个 expert 的 w1、w3 及对应 scales | 是 | 配对 shards 到齐后统一 scale requantize |
| MoE w2 | 一个 expert 的 w2 及 scale | 否 | shard 到达后直接处理 serving slice |
| 全层 activation scale | scale 定义覆盖的全部 expert/layer 贡献 | 是 | 依赖集合完成，通常在 `FINISH` |
| MLA derived `W_UK_T/W_UV` | 所有 base tensor 已更新的 MLA layer | 不属于传输 buffer | `refresh_derived_state()` |

如果一个量化方法的 scale/layout 依赖整个 layer，不能为了省 buffer 强行拆成 expert unit；该方法延迟到 `FINISH` 处理。

#### 后端适用边界

quant method 只有在 serving 值能够由一个 unit 独立确定时才声明细粒度 unit。以 per-tensor FP8 MoE 为例，Triton 和 batched-Triton 在非 FNUZ 平台上可以按 expert 完成 `w13` requantization；需要 layer-wide shuffle、padding 或 backend-owned conversion 的 FlashInfer、Marlin、AITER 和 DeepGEMM 路径不得声明相同粒度，而应回退到 `FINISH` 的 selective PWAL。是否启用 eager unit 必须由 quant method/backend capability 决定，不能仅根据 dtype 推断。

per-expert FP8 的参考 unit 如下：

| Unit | Required keys | Staged values | Finalize |
|---|---|---|---|
| Per-tensor `w13[e]` | expert e 的 w1/w3 weights 和 scales | `[2I, H]` weight、`[2]` scale | 使用 `max(s1, s3)` requantize 两半并写入 `w13[e]` |
| Block `w13[e]` | expert e 的 w1/w3 weights 和 block scales | 无 | checkpoint format 已是 serving format，loader 直接写入 |
| `w2[e]` | expert e 的 w2 weight 和 scale | 无 | 非 fused 参数，loader 直接写入 |
| Static `input_scale` | 所有 expert 的 input scales | 每个 expert 一个标量 | 在 `FINISH` 求全层 max |

unit finalize 应与 cold load 复用同一个量化函数，使 streaming reload 与冷启动转换保持 bit-identical。

### 2.4 分片传输协议

由于一次传输只携带一个 sharding，传输 record 必须携带足够的逻辑身份，而不能只携带参数名：

```text
Record {
  update_id, sequence_no,
  tensor_name, logical_unit_id,
  tp/ep/expert/shard coordinates,
  byte_range, shape, stride, dtype,
  quant_format/version, checksum
}
```

发送端可以把 records 打包成受 `max_chunk_bytes` 和显存预算限制的 bucket；bucket 可以跨 parameter、expert 和 layer。接收端按 record 坐标写入对应 unit，不把 bucket 当作完成边界。乱序、重复 record 必须可去重；缺失 sharding 在 `FINISH` 时报告具体 unit/key。

这吸收了 SGLang issue 32335 关于分片传输降低峰值内存的思路，但结合 vLLM 的 reload 流程后，chunk 只承担通信/调度职责，unit 才承担量化/提交职责：

```text
transport bucket -> model.load_weights(record)
                 -> unit tracker
                 -> unit complete: quantize + publish + free buffer
                 -> FINISH: verify all units + remaining PWAL/derived
```

接收端实施 backpressure：当 quant tracker 持有的 staging bytes 达到配置预算时，暂停发送或减小 bucket；优先补齐接近完成的 unit，使其量化并释放 buffer。NCCL、CUDA IPC、文件传输共享 record schema，重传键为 `(update_id, sequence_no)`。

### 2.5 一致性与失败语义

reload 开始前已经 drain 请求，因此 eager unit 写入不会被正在运行的 forward 观察到。“立即量化并释放 buffer”是 update 窗口内的内存优化，不是 hot-swap 或逐 unit 对外发布。只有 `FINISH` 完成所有 coverage、selective PWAL、derived refresh 和 manifest check 后才 resume 请求。

in-place 更新不能廉价回滚已经写入的 unit。`FINISH` 发现缺失 shard、转换失败或 OOM 时，worker 不得继续 serving；控制面应补齐同一 update，或重建/重新加载 worker。若未来要求失败后继续用旧权重，则必须引入完整 shadow copy/双 buffer，这不属于本 RFC 的 v1/v2 scope。

## 3. 验证要求

每个量化后端至少验证：乱序和跨 bucket sharding、重复/重传、缺失 key、dtype mismatch、unit buffer 释放、cold-load 与 reload 的 tensor checksum/输出一致性，以及 Parameter identity 和 `data_ptr` 不变。v1/v2 矩阵中的“支持”只在对应测试通过后声明。

参考实现已在 H200 上使用 Qwen3-30B-A3B 的 2-layer、128-expert per-tensor FP8 配置验证。每层发现 256 个 units；reload 后 serving tensors 与 checkpoint B 的 cold load 按 parameter SHA256 bit-identical，Parameter identity 和 storage address 变化数为 0，生成 token 与 cold-served B 一致且不同于 A。streaming 路径的 reload peak extra memory 为 1256 MiB，关闭 streaming 后为 2409 MiB。

同一 checkpoint pair 还通过 vLLM NCCL weight-transfer engine 验证：trainer 在第二张 GPU 上使用 `vllm-rl-day0-kit` publisher，将 1567 个 tensors 放入 5 个 packed buckets。一个 expert 的 shards 可以跨 bucket，服务端仍基于 key-set coverage 完成 2 层共 512 个 units，更新后的输出与 cold-served B 一致。该结果验证了 transport bucket 不是 completion boundary，也验证了 unit 完成后释放 expert staging 可以降低峰值显存。
