# Prefix-Chain Fracture Eviction 

## 1. 研究动机

现代 LLM serving 系统会在多个请求共享相同 prompt prefix 时复用 KV-cache blocks。该优化可以减少重复 prefill 计算，从而降低 Time-To-First-Token，提升吞吐量并降低 serving 成本。然而，在 multi-tenant 场景中，cache 不再只是单个用户的本地优化，而是一个跨用户共享的系统状态。即使攻击者不能读取或覆盖 KV tensor，也可能通过正常 API 请求影响 cache block 的驻留状态。

已有 KV-cache 安全研究主要关注以下方向：

- **Timing side-channel**：API-only 攻击者通过测量 TTFT 推断某个 prefix 是否已经被缓存。
- **Hash 或 serialization collision**：错误的 cache hit 导致 poisoned retrieval 或内容绕过。
- **Direct KV access 或 corruption**：攻击者读取、重构、覆盖或 bit-flip KV-cache 内容。

本项目探索一个不同的攻击面：

> 一个 co-tenant 不直接读取或修改 KV-cache 内容，而是操纵共享 prefix blocks 的生命周期，使某个 branch-entry block 更可能被 evict。由于 vLLM 的 prefix lookup 是链式的，evict 一个较早的 branch block 可能导致后续更长的 suffix 无法复用。

我们将该现象称为 **Prefix-Chain Fracture Eviction**。

---

## 2. 核心洞察

假设 victim 的 prompt 结构为：

```text
P = G + B + S + q_v
```

其中：

- `G` 是全局共享 root，例如 system prompt。
- `B` 是 branch-specific template，例如 agent workflow、tool schema header、RAG template 或 enterprise task template。
- `S` 是较长且可复用的 suffix，例如 examples、schema definitions、tool descriptions 或 documents。
- `q_v` 是 victim-specific query。

攻击者发送只刷新 `G`、但不刷新 `B` 的请求：

```text
A_refresh = G + q_a
```

或者：

```text
A_refresh = G + B' + q_a, where B' != B
```

这样会形成一种相对 aging 效应：

```text
G remains young
B becomes old
```

如果后续一次较小的 allocation 或 background workload evict 掉 `B` 的第一个 block，那么 victim 下一次请求会出现：

```text
G hit
B miss
S unusable
```

原因是 vLLM 的 cache lookup 是 prefix-chained。一旦 lookup 在 `B` 处 miss，系统就不能继续把 `S` 中的后续 blocks 作为同一个连续 prefix 的一部分复用，即使它们的物理 blocks 可能仍然存在于 cache 中。

因此，该攻击不是 brute-force cache flooding，而是一种 **branch-level lifecycle attack**：

```text
keep the shared root alive
let the branch age
trigger a branch-entry fracture
force recomputation of the downstream suffix
```

---

## 3. 为什么 vLLM 是合适的实验平台

该 idea 来自 vLLM prefix caching 的几个关键性质：

1. **Block-based KV cache**：vLLM 将每个请求的 KV cache 划分为固定大小的 blocks。
2. **Chained hash lookup**：一个 block 的 hash 依赖 parent hash、当前 block tokens，以及额外 hash，例如 LoRA ID、multimodal hash 或 cache salt。
3. **Full-block caching**：只有 full blocks 会被可靠缓存和复用。
4. **Touch on hit**：当一个 cached block 被复用时，它会被 touch，reference count 增加，并在 active 期间受到 eviction 保护。
5. **LRU free queue eviction**：新 block allocation 会从 free queue head 弹出 block；如果该 block 已经 cached，则会被 evict。
6. **First-miss termination**：prefix lookup 从第一个 block 向后顺序进行；一旦某个 block miss，后续 blocks 将重新分配。

这些性质使 branch-entry block 成为高杠杆攻击目标。攻击者不需要 evict 整个 prompt，只需要使一个战略位置的 branch block miss，就可能 fracture 后续 suffix 的复用。

---

## 4. Threat Model

### Attacker

攻击者是普通 co-tenant，可以：

- 向同一个 LLM serving 系统发送 API 请求。
- 知道或猜测共享 root prompt `G`。
- 知道或猜测公开或半公开的 branch template `B`，例如开源 agent template、tool schema 或 service workflow。
- 观察自己的 latency，尤其是 TTFT。
- 可选地在同一 cache-sharing group 中创建多个 session。

攻击者不能：

- 读取 victim prompts。
- 读取或写入 KV-cache tensors。
- 访问 vLLM 内部 metadata，例如 block ID、block hash、reference count 或 free-queue rank。
- 修改模型权重、tokenizer、scheduler 或 serving code。
- 依赖 hash collision 或硬件 fault。

### Victim

victim 提交共享相同 root 和 branch 的请求：

```text
P_v = G + B + S + q_v
```

在正常情况下，victim 会受益于 `G + B + S` 的 prefix cache reuse。

### Deployment Assumption

该攻击与以下部署场景相关：

- 没有 per-user isolation 的 global prefix caching。
- 使用 organization-level 或 application-level cache salt。
- 共享 system prompt caching。
- 多租户 serving 中存在 common agent、RAG 或 tool templates。

---

## 5. 研究问题

本项目研究：

> 一个 co-tenant 是否能在不进行 brute-force cache flooding 的情况下造成 branch-level prefix-cache fracture，并使 victim 请求产生不成比例的 recomputation？

更具体地说：

1. 攻击者是否能保持 root blocks `G` 可复用，同时使 branch-entry blocks `B` 更容易被 evict？
2. branch-entry block 的 eviction 是否会导致 victim prefix hit length 出现明显下降？
3. 攻击造成的 TTFT increase 是否显著高于攻击者请求成本所能解释的影响？
4. 与 naive cache flooding 相比，该攻击是否表现出更低 request volume 和更强 branch-selective effect？

---

## 6. Attack Design

### 6.1 Root Refresh

攻击者发送低成本请求，只共享 root：

```text
G + q_a
```

或者共享另一个 branch：

```text
G + B' + q_a
```

预期效果：

```text
HitRate(G) remains high
Age(G) remains low
B is not touched
Age(B) increases
```

### 6.2 Branch Aging

攻击者不访问 victim branch `B`。在正常 background workload 下，`B` 相对于 `G` 逐渐变旧。

预期效果：

```text
Age(G) < Age(B)
```

### 6.3 Minimal Trigger Allocation

攻击者不进行大规模 flooding，而是由攻击者或自然 background workload 触发少量新 block allocation。由于 `B` 比 `G` 更旧，branch-entry block 更可能靠近 LRU eviction frontier。

预期 fracture：

```text
G hit
B_1 evicted or missed
S not reused
```

### 6.4 Optional Ref-Count Shielding

更强的变体是保持少量 root-sharing requests 处于 active 状态，使 root blocks 具有正 reference count，从而暂时将它们移出 eviction candidate set：

```text
ref_cnt(G) > 0
ref_cnt(B) = 0
```

该策略在实验中需要谨慎使用，避免退化成通用 resource-exhaustion attack。

---

## 7. 对 Multi-Tenant Serving 的影响

### 7.1 跨租户性能干扰

victim 原本应复用长 prefix，但攻击后被迫重新计算 `B + S`。

预期症状：

```text
TTFT_v increases
Prefill tokens increase
Throughput decreases
GPU prefill load increases
```

### 7.2 服务公平性下降

攻击者只承担刷新 `G` 的成本，而 victim 承担重新计算 `B + S` 的成本。

定义：

```text
Fracture Amplification Ratio (FAR)
= victim recomputed tokens / attacker refresh-or-trigger tokens
```

较高 FAR 表明低成本 lifecycle manipulation 可以对其他租户施加不成比例的计算成本。

### 7.3 Coarse Cache Salting 的不完全保护

如果 cache salt 按用户或 session 粒度设置，则攻击应被阻断。但如果 salt 按 organization、project、application 或 trust group 粒度设置，那么同一 salt group 内互不信任的用户仍然可能相互干扰 cache lifecycle。

---

## 8. Experimental Plan

实验评估建议分为两个阶段。

### Stage 1: White-Box Mechanism Validation

目的：

> 验证 `G` 仍然 cached，但 `B` 被 evict，导致 `S` 无法复用。

在 vLLM 的 KV-cache manager 中加入日志，记录：

```text
request_id
block_id
block_hash
logical_position
ref_cnt
in_free_queue
free_queue_rank
event_type: hit | touch | allocate | free | evict
prefix_hit_len
```

建议记录位置：

- `get_computed_blocks()`
  - 记录 prefix hit length。
  - 记录命中的 logical blocks。
- `allocate_slots()`
  - 记录被 touch 的 computed blocks。
  - 记录从 free queue 中弹出的 blocks。
  - 记录被 evict 的 cached blocks。
- request free path
  - 记录 blocks 何时返回 free queue。

#### Prompt Construction

构造：

```text
G = shared root prompt
B = branch-specific workflow/template
S = long reusable suffix
q_v = victim query
q_a = attacker query
```

设计要求：

- 将 `G`、`B` 和 `S` 对齐到 full block 边界。
- 让 `B` 的第一个 token 位于一个 block 的开头。
- 使 `S` 足够长，例如 512、1024、2048 或 4096 tokens。
- 避免把目标 block 放在 prefix-suffix boundary，因为 boundary block 可能包含 request-specific tokens，不一定能可靠跨请求复用。

示例受控长度：

```text
G = 256 tokens
B = 128 tokens
S = 2048 tokens
block_size = 16 tokens
```

#### Clean Baseline

1. Warm up：

```text
G + B + S + q_v
```

2. 使用不同 victim query 测试相同 prefix：

```text
G + B + S + q'_v
```

预期：

```text
HitLen_clean ≈ blocks(G + B + S)
TTFT_clean is low
```

#### Attack Condition

1. Warm up：

```text
G + B + S + q_v
```

2. Refresh root：

```text
G + q_a
```

或：

```text
G + B' + q_a
```

3. 让 background workload 或少量 trigger allocation 发生。
4. 再次测试 victim：

```text
G + B + S + q'_v
```

预期：

```text
G blocks hit
B_1 block missed or evicted
S blocks not reused
HitLen_attack ≈ blocks(G)
TTFT_attack > TTFT_clean
```

---

### Stage 2: Black-Box Attack Observability

目的：

> 证明攻击效果可以通过攻击者可见的 TTFT 被观测，而不需要内部 metadata。

攻击者 probe：

```text
G + q_a
G + B + q_a
G + B + S_short + q_a
```

成功 fracture 后预期：

```text
TTFT(G + q_a) remains close to warm-cache latency
TTFT(G + B + q_a) increases
TTFT(G + B + S_short + q_a) increases further
```

这说明：

```text
G hit
B miss
downstream suffix recomputed
```

由于 TTFT 存在噪声，建议使用 repeated trials 和 median TTFT。

---

## 9. Workload Design

至少比较四类 workloads。


| Workload                    | Description                                  | Expected Result                               |
| --------------------------- | -------------------------------------------- | --------------------------------------------- |
| No Attack                   | Victim 重复使用`G+B+S`                       | High hit length, low TTFT                     |
| Random Background           | 随机 multi-tenant prompts                    | 一定程度 eviction，但 fracture point 不稳定   |
| Naive Flooding              | 大量 unique prompts                          | 退化明显，但攻击成本高                        |
| Root Refresh + Branch Aging | Refresh`G`，age `B`，少量 trigger allocation | 更高 branch miss rate，且 request volume 更低 |

主要对比对象是 naive flooding。该攻击是否成立，关键在于它能否以更少请求或更低 token 成本实现 branch-selective degradation。

---

## 10. Metrics

### 10.1 Prefix Hit Length

```text
HitLen_v = number of prefix blocks reused by victim
```

预期：

```text
HitLen_clean ≈ blocks(G+B+S)
HitLen_attack ≈ blocks(G)
```

### 10.2 Branch Miss Rate

```text
BMR = Pr[G hit AND B miss]
```

这是衡量 branch-level fracture 的核心指标。

### 10.3 Root Survival Rate

```text
RSR = Pr[G hit after attack]
```

高 RSR 可以区分 branch fracture 和 full-cache eviction。

### 10.4 TTFT Increase

```text
ΔTTFT = TTFT_attack - TTFT_clean
```

### 10.5 Recomputed Tokens

```text
RecomputedTokens = blocks(B+S) × block_size
```

### 10.6 Fracture Amplification Ratio

```text
FAR = RecomputedTokens_v / AttackerTokens
```

高 FAR 说明攻击成本与 victim recomputation cost 之间存在明显不对称。

### 10.7 Attack Request Volume

```text
AttackRequests = number of attacker requests before fracture
```

它应显著低于 naive flooding。

---

## 11. Expected Experimental Figures

1. **HitLen before and after attack**

```text
x-axis: workload
y-axis: victim prefix hit length
```

2. **Branch Miss Rate**

```text
x-axis: workload
y-axis: BMR
```

3. **TTFT Increase**

```text
x-axis: suffix length |S|
y-axis: ΔTTFT
```

4. **Fracture Amplification Ratio**

```text
x-axis: workload
y-axis: FAR
```

5. **Root Survival vs. Branch Eviction**

```text
x-axis: time or request index
y-axis: hit/miss status of G and B
```

---

## 12. Success Criteria

如果观察到以下现象，则 idea 得到支持：

```text
HitRate(G) remains high
EvictRate(B_1) increases under Root Refresh + Branch Aging
HitLen_v drops from blocks(G+B+S) to approximately blocks(G)
TTFT_v increases after fracture
FAR is higher than random background and naive flooding
AttackRequests is lower than naive flooding
```

最强证据是：

```text
G hit, B miss, S unusable
```

---

## 13. Failure Modes and Interpretation

### B 没有被 evict

可能原因：

```text
B 本身过于热门，并且被频繁 touch。
```

解决方法：

```text
使用更 realistic 的 branch，其中 G 高频，但 B 中频或低频。
```

### G 也被 evict

可能原因：

```text
Trigger allocation 太强，使方法退化为 flooding。
```

解决方法：

```text
降低 trigger pressure，或加入轻量 root refresh/ref-count shielding。
```

### TTFT 差异较弱

可能原因：

```text
S 太短。
```

解决方法：

```text
将 S 增加到 1024、2048 或 4096 tokens。
```

### Black-box probing 噪声较大

可能原因：

```text
TTFT 受到 scheduling 和 batching 噪声影响。
```

解决方法：

```text
使用 deterministic decoding、固定 max_tokens、受控 concurrency，并对多次实验取 median。
```

---

## 14. Defense Directions

潜在防御方向包括：

1. **Branch-aware eviction**

   - 保护 downstream suffix recomputation cost 较高的 branch-entry blocks。
2. **Root/branch reuse accounting**

   - 分别统计 global root 和 branch entry 的 reuse，而不是统一处理所有 cached blocks。
3. **Minimum residency guarantee**

   - 为最近创建的 branch-entry blocks 提供最小驻留时间。
4. **Fracture-aware replacement**

   - 在 evict 某个 block 前估计其可能造成的 recomputation damage。
5. **Runtime anomaly detection**

   - 检测某个 tenant 反复刷新 common root，而其他 unrelated branches 持续 aging out 的模式。

---

## 15. 与已有 KV-Cache Attacks 的区别


| Attack Type              | Attacker Capability                   | Main Goal              | Difference from This Project                           |
| ------------------------ | ------------------------------------- | ---------------------- | ------------------------------------------------------ |
| Timing side channel      | API-only, observes TTFT               | 推断 cached prompts    | 本项目利用 cache lifecycle interference 降低复用效率   |
| Hash collision poisoning | API-only plus collision construction  | 强制错误 cache hit     | 本项目不依赖 hash collision                            |
| Bit-flip corruption      | Hardware fault or fault injection     | 污染 shared KV content | 本项目不修改 KV values                                 |
| KV reconstruction        | Access to KV tensors or model weights | 恢复用户输入           | 本项目假设无 KV access                                 |
| History swapping         | Read/write KV tensors                 | 控制生成方向           | 本项目改变 cache residency，而不是 cache content       |
| Naive cache flooding     | Many unique requests                  | 全局 cache degradation | 本项目以更低 request volume 触发 branch-entry fracture |

---

## 16. Minimal Reproduction Checklist

- [ ]  安装 vLLM，并开启 prefix caching。
- [ ]  使用小模型快速迭代，例如 Qwen2.5-1.5B 或 Llama-3.2-1B。
- [ ]  设置受限 KV-cache capacity，便于观察 controlled eviction。
- [ ]  构造对齐的 `G`、`B` 和 `S` prompts。
- [ ]  在 `get_computed_blocks()` 和 `allocate_slots()` 添加 white-box logging。
- [ ]  测量 clean baseline 的 hit length 和 TTFT。
- [ ]  运行 Root Refresh + Branch Aging。
- [ ]  添加少量 trigger allocation 或 controlled background workload。
- [ ]  再次测量 victim hit length 和 TTFT。
- [ ]  与 no attack、random background 和 naive flooding 对比。
- [ ]  报告 BMR、RSR、HitLen、ΔTTFT、FAR 和 attack request volume。

---

## 17. Summary

Prefix-Chain Fracture Eviction 研究一种细粒度的 multi-tenant KV-cache 风险：

```text
Root remains cached.
Branch entry is evicted.
Downstream suffix cannot be reused.
Victim recomputes a long prefix.
```

核心 claim 是：

> 一个 co-tenant 不需要覆盖 KV tensors、不需要利用 hash collisions，也不需要 flood 整个 cache。通过选择性刷新 shared root，并让较低频 branch aging，攻击者可以使 vLLM 的 lifecycle policy 更倾向于 branch-entry fracture，从而对 victim 造成不成比例的 recomputation。

因此，该问题本质上是 **cache-lifecycle interference**，而不是直接的 cache-content attack。
