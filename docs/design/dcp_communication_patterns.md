# DCP Communication Patterns

This document describes the communication patterns for Decode Context Parallelism (DCP) with various configurations of Tensor Parallelism (TP) and Prefill Context Parallelism (PCP).

## Background

- **TP (Tensor Parallelism)**: Splits attention heads across ranks. Each rank has `H/TP` heads.
- **PCP (Prefill Context Parallelism)**: Splits prefill tokens across ranks. Each PCP slice has its own TP group.
- **DCP (Decode Context Parallelism)**: Splits KV cache context across ranks for decode.

### Rank Layout

Ranks are laid out as `(ep, dp, pp, pcp, tp)`. For simplicity, we assume `ep=dp=pp=1`.

For **PCP=2, TP=4** (8 ranks):

```text
        TP=0  TP=1  TP=2  TP=3
PCP=0    0     1     2     3
PCP=1    4     5     6     7
```

For **PCP=2, TP=2** (4 ranks):

```text
        TP=0  TP=1
PCP=0    0     1
PCP=1    2     3
```

For **PCP=1, TP=4** (4 ranks):

```text
        TP=0  TP=1  TP=2  TP=3
PCP=0    0     1     2     3
```

### DCP Group Formation

DCP groups are formed by spanning **PCP first, then TP**:

1. Transpose layout to `(tp, pcp)`
2. Flatten and reshape to `(-1, dcp_size)`

---

## Case 1: PCP=1, TP=4, DCP=4

**Groups:**

| Group Type | Ranks |
| ------------ | ------- |
| TP group | `[0, 1, 2, 3]` |
| DCP group | `[0, 1, 2, 3]` (same as TP) |

**Head distribution:**

| Rank | TP Position | Heads |
| ------ | ------------- | ------- |
| 0 | 0 | `[0, H/4)` |
| 1 | 1 | `[H/4, H/2)` |
| 2 | 2 | `[H/2, 3H/4)` |
| 3 | 3 | `[3H/4, H)` |

**DCP Decode Communication:**

```text
Step 1: TP All-Gather (query)
┌─────────────────────────────────────────────────────┐
│  All ranks gather with TP group [0,1,2,3]           │
│  Each rank: H/4 heads → H heads                     │
└─────────────────────────────────────────────────────┘

Step 2: Attention
┌─────────────────────────────────────────────────────┐
│  Each rank computes attention with ALL H heads      │
│  against its local KV slice (1/4 of context)        │
└─────────────────────────────────────────────────────┘

Step 3: DCP Reduce (reduce-scatter)
┌─────────────────────────────────────────────────────┐
│  DCP == TP, so reduce-scatter works directly        │
│  Each rank gets its original H/4 heads back         │
│  Rank 0: heads [0, H/4)                             │
│  Rank 1: heads [H/4, H/2)                           │
│  Rank 2: heads [H/2, 3H/4)                          │
│  Rank 3: heads [3H/4, H)                            │
└─────────────────────────────────────────────────────┘
```

**Optimization:** None needed. DCP == TP is optimal.

---

## Case 2: PCP=1, TP=4, DCP=2

**Groups:**

| Group Type | Ranks |
| ------------ | ------- |
| TP group | `[0, 1, 2, 3]` |
| DCP groups | `[0, 1]`, `[2, 3]` |

**Head distribution:**

| Rank | TP Position | DCP Group | Heads |
| ------ | ------------- | ----------- | ------- |
| 0 | 0 | 0 | `[0, H/4)` |
| 1 | 1 | 0 | `[H/4, H/2)` |
| 2 | 2 | 1 | `[H/2, 3H/4)` |
| 3 | 3 | 1 | `[3H/4, H)` |

**DCP Decode Communication:**

```text
Step 1: TP All-Gather (query)
┌─────────────────────────────────────────────────────┐
│  All ranks gather with TP group [0,1,2,3]           │
│  Each rank: H/4 heads → H heads                     │
│  (Gathers more than needed!)                        │
└─────────────────────────────────────────────────────┘

Step 2: Attention
┌─────────────────────────────────────────────────────┐
│  DCP group [0,1]: each computes with H heads        │
│    against 1/2 of context                           │
│  DCP group [2,3]: each computes with H heads        │
│    against 1/2 of context                           │
└─────────────────────────────────────────────────────┘

Step 3: DCP Reduce (all-reduce + slice)
┌─────────────────────────────────────────────────────┐
│  DCP ⊂ TP, so need all-reduce + manual slice        │
│  Within [0,1]: all-reduce, then each slices to H/4  │
│  Within [2,3]: all-reduce, then each slices to H/4  │
└─────────────────────────────────────────────────────┘
```

**Optimization:** Use partial-TP all-gather

- DCP group `[0,1]` covers TP positions `{0, 1}` → only need `H/2` heads
- DCP group `[2,3]` covers TP positions `{2, 3}` → only need `H/2` heads
- Partial-TP groups: `[0,1]` and `[2,3]` (same as DCP groups in this case)

```text
Optimized Step 1: Partial-TP All-Gather
┌─────────────────────────────────────────────────────┐
│  Ranks [0,1] gather within [0,1]: H/4 → H/2 heads   │
│  Ranks [2,3] gather within [2,3]: H/4 → H/2 heads   │
│  (Half the communication!)                          │
└─────────────────────────────────────────────────────┘
```

---

## Case 3: PCP=2, TP=2, DCP=4

**Groups:**

| Group Type | Ranks |
| ------------ | ------- |
| TP groups | `[0, 1]` (PCP=0), `[2, 3]` (PCP=1) |
| DCP group | `[0, 2, 1, 3]` (all ranks) |
| PCP group | `[0, 1, 2, 3]` |

**Head distribution:**

| Rank | TP Position | PCP Slice | Heads |
| ------ | ------------- | ----------- | ------- |
| 0 | 0 | 0 | `[0, H/2)` |
| 1 | 1 | 0 | `[H/2, H)` |
| 2 | 0 | 1 | `[0, H/2)` |
| 3 | 1 | 1 | `[H/2, H)` |

**DCP Decode Communication:**

```text
Step 1: TP All-Gather (query)
┌─────────────────────────────────────────────────────┐
│  Rank 0 gathers with [0,1] → H heads                │
│  Rank 1 gathers with [0,1] → H heads                │
│  Rank 2 gathers with [2,3] → H heads                │
│  Rank 3 gathers with [2,3] → H heads                │
└─────────────────────────────────────────────────────┘

Step 2: Attention
┌─────────────────────────────────────────────────────┐
│  Each rank computes with ALL H heads                │
│  against its local KV slice (1/4 of context)        │
└─────────────────────────────────────────────────────┘

Step 3: DCP Reduce (all-reduce + slice)
┌─────────────────────────────────────────────────────┐
│  DCP=4 > TP=2, so need all-reduce + slice           │
│  All-reduce across [0,2,1,3]                        │
│  Each rank slices to its TP-local H/2 heads         │
└─────────────────────────────────────────────────────┘
```

**Note:** DCP spans both PCP and TP dimensions. All ranks are in one DCP group.

---

## Case 4: PCP=2, TP=2, DCP=2

**Groups:**

| Group Type | Ranks |
| ------------ | ------- |
| TP groups | `[0, 1]` (PCP=0), `[2, 3]` (PCP=1) |
| DCP groups | `[0, 2]`, `[1, 3]` (span PCP, same TP!) |

**Head distribution:**

| Rank | TP Position | PCP Slice | DCP Group | Heads |
| ------ | ------------- | ----------- | ----------- | ------- |
| 0 | 0 | 0 | 0 | `[0, H/2)` |
| 2 | 0 | 1 | 0 | `[0, H/2)` |
| 1 | 1 | 0 | 1 | `[H/2, H)` |
| 3 | 1 | 1 | 1 | `[H/2, H)` |

**Key insight:** Ranks in the same DCP group have the **same TP position** (same heads)!

**DCP Decode Communication:**

```text
Step 1: TP All-Gather (query)
┌─────────────────────────────────────────────────────┐
│  Rank 0 gathers with [0,1] → H heads                │
│  Rank 2 gathers with [2,3] → H heads                │
│  (Both DCP peers do redundant work!)                │
└─────────────────────────────────────────────────────┘

Step 2: Attention
┌─────────────────────────────────────────────────────┐
│  DCP group [0,2]: each computes with H heads        │
│    Rank 0: against KV slice A                       │
│    Rank 2: against KV slice B                       │
└─────────────────────────────────────────────────────┘

Step 3: DCP Reduce (all-reduce, no scatter needed)
┌─────────────────────────────────────────────────────┐
│  All-reduce within [0,2] and [1,3]                  │
│  Each rank keeps its original H/2 heads             │
│  (No redistribution needed - same TP position!)    │
└─────────────────────────────────────────────────────┘
```

**Optimization:** Skip TP all-gather entirely!

- Ranks 0 and 2 already have the same heads `[0, H/2)`
- They can compute attention with `H/2` heads directly
- Partial-TP group size = 1 (no all-gather needed)

```text
Optimized Step 1: No All-Gather!
┌─────────────────────────────────────────────────────┐
│  Each rank keeps its original H/2 heads             │
│  (Zero communication!)                              │
└─────────────────────────────────────────────────────┘
```

---

## Case 5: PCP=2, TP=4, DCP=4

**Groups:**

| Group Type | Ranks |
| ------------ | ------- |
| TP groups | `[0,1,2,3]` (PCP=0), `[4,5,6,7]` (PCP=1) |
| DCP groups | `[0,4,1,5]`, `[2,6,3,7]` |

**Head distribution:**

| Rank | TP Position | PCP Slice | DCP Group | Heads |
| ------ | ------------- | ----------- | ----------- | ------- |
| 0 | 0 | 0 | 0 | `[0, H/4)` |
| 4 | 0 | 1 | 0 | `[0, H/4)` |
| 1 | 1 | 0 | 0 | `[H/4, H/2)` |
| 5 | 1 | 1 | 0 | `[H/4, H/2)` |
| 2 | 2 | 0 | 1 | `[H/2, 3H/4)` |
| 6 | 2 | 1 | 1 | `[H/2, 3H/4)` |
| 3 | 3 | 0 | 1 | `[3H/4, H)` |
| 7 | 3 | 1 | 1 | `[3H/4, H)` |

**Key insight:** Each DCP group covers **2 unique TP positions** (half of TP=4).

**DCP Decode Communication (current):**

```text
Step 1: TP All-Gather (query)
┌─────────────────────────────────────────────────────┐
│  Rank 0 gathers with [0,1,2,3] → H heads            │
│  Rank 4 gathers with [4,5,6,7] → H heads            │
│  (Gathers H heads but only needs H/2!)              │
└─────────────────────────────────────────────────────┘

Step 2: Attention
┌─────────────────────────────────────────────────────┐
│  Each rank computes with ALL H heads                │
│  against its local KV slice (1/4 of context)        │
└─────────────────────────────────────────────────────┘

Step 3: DCP Reduce (all-reduce + slice)
┌─────────────────────────────────────────────────────┐
│  DCP=4, TP=4, but PCP>1 so can't reduce-scatter     │
│  All-reduce within DCP group, slice to H/4 heads    │
└─────────────────────────────────────────────────────┘
```

**Optimization:** Use partial-TP all-gather

DCP group analysis:

- `[0,4,1,5]` covers TP positions `{0, 1}` → needs `H/2` heads
- `[2,6,3,7]` covers TP positions `{2, 3}` → needs `H/2` heads

Partial-TP groups (per PCP slice):

| Partial-TP Group | Ranks | TP Positions | For DCP Group |
| ------------------ | ------- | -------------- | --------------- |
| `[0, 1]` | PCP=0 | {0, 1} | 0 |
| `[4, 5]` | PCP=1 | {0, 1} | 0 |
| `[2, 3]` | PCP=0 | {2, 3} | 1 |
| `[6, 7]` | PCP=1 | {2, 3} | 1 |

```text
Optimized Step 1: Partial-TP All-Gather
┌─────────────────────────────────────────────────────┐
│  Ranks [0,1] gather within [0,1]: H/4 → H/2 heads   │
│  Ranks [4,5] gather within [4,5]: H/4 → H/2 heads   │
│  Ranks [2,3] gather within [2,3]: H/4 → H/2 heads   │
│  Ranks [6,7] gather within [6,7]: H/4 → H/2 heads   │
│  (Half the communication vs full TP all-gather!)   │
└─────────────────────────────────────────────────────┘
```

---

## Summary: When to Use Each Pattern

| Condition | All-Gather | Reduce |
| ----------- | ------------ | -------- |
| `DCP == TP` and `PCP == 1` | Full TP | reduce-scatter |
| `DCP < TP` and `PCP == 1` | Partial-TP (DCP group) | all-reduce + slice |
| `DCP == TP * PCP` | Full TP | all-reduce + slice |
| `DCP < TP * PCP` and `PCP > 1` | Partial-TP | all-reduce + slice |

### Partial-TP Group Formula

```python
unique_tp_per_dcp = dcp_size // pcp_size
num_dcp_groups = (tp_size * pcp_size) // dcp_size

# For each DCP group i, for each PCP slice p:
#   Partial-TP group = ranks at TP positions [i * unique_tp_per_dcp, (i+1) * unique_tp_per_dcp)
#                      within PCP slice p
```

### Communication Savings

| Config | Full TP All-Gather | Partial-TP All-Gather | Savings |
| -------- | ------------------- | ---------------------- | --------- |
| PCP=1, TP=4, DCP=4 | H | H | 0% |
| PCP=1, TP=4, DCP=2 | H | H/2 | 50% |
| PCP=2, TP=2, DCP=2 | H | 0 (skip!) | 100% |
| PCP=2, TP=4, DCP=4 | H | H/2 | 50% |

---

## PD Disaggregation: KV Cache Coverage and rank matching

In disaggregated prefill/decode, the **prefill instance** (configured with `pcp_size_P`,
`tp_size_P`, `dcp_size_P`) and the **decode instance** (configured with `pcp_size_D`,
`tp_size_D`, `dcp_size_D`) are separate processes. The prefill instance must transfer
its locally-stored KV cache to the correct ranks on the decode side.

### KV Cache Storage

**Only DCP shards the KV cache.** PCP splits token computation during prefill, but
after `pcp_kv_allgather_and_restore` each rank has the full sequence KV. Only the
tokens assigned by `dcp_local_rank` are then written to the local cache:

```text
token position i is stored on the rank where:
    (i // interleave_size) % dcp_size == dcp_local_rank
```

A rank's `dcp_local_rank` is derived from the DCP group formation:

```python
flat_idx       = tp_rank * pcp_size + pcp_rank
dcp_local_rank = flat_idx % dcp_size
```

Therefore, each rank's local KV cache holds:

| Dimension | Range |
| ----------- | ------- |
| **Heads** | `[tp_rank * H_kv / tp_size, (tp_rank + 1) * H_kv / tp_size)` |
| **Tokens** | `{ i : (i // interleave_size) % dcp_size == dcp_local_rank }` |

where `H_kv` is the total number of KV heads (GQA: `H_kv ≤ H_q`).

### GQA and KV Replication

When `tp_size >= H_kv` (more TP ranks than KV heads), KV is replicated across
`tp_size / H_kv` TP ranks per KV head group. These replica ranks hold **identical**
KV heads. In this case the maximum useful DCP size is:

```text
DCP_max = PCP * TP / H_kv
```

because that is exactly the number of ranks that share the same KV head group across
both the TP-replica and PCP dimensions, yet each holds a **different** token slice.

The **effective TP for KV coverage** is therefore:

```python
eff_tp = min(tp_size, H_kv)          # number of distinct KV head groups
kv_group = tp_rank * eff_tp // tp_size  # KV head group index in [0, eff_tp)
```

### Coverage Condition

A prefill rank `r_P` and a decode rank `r_D` have overlapping KV data iff both
conditions hold simultaneously:

**Condition H – KV head overlap:**

```text
kv_group_P / eff_tp_P  <  (kv_group_D + 1) / eff_tp_D
    AND
kv_group_D / eff_tp_D  <  (kv_group_P + 1) / eff_tp_P
```

where:

```python
eff_tp_P  = min(tp_size_P, H_kv)
eff_tp_D  = min(tp_size_D, H_kv)
kv_group_P = tp_rank_P * eff_tp_P // tp_size_P
kv_group_D = tp_rank_D * eff_tp_D // tp_size_D
```

When `tp_size >= H_kv` on both sides this simplifies to `kv_group_P == kv_group_D`.

**Condition T – token (interleave) overlap:**

```python
g = gcd(dcp_size_P, dcp_size_D)
dcp_local_rank_P % g  ==  dcp_local_rank_D % g
```

Because `g | dcp_size_P` and `g | dcp_size_D`, this is equivalent to:

```python
flat_idx_P % g  ==  flat_idx_D % g
# where flat_idx = tp_rank * pcp_size + pcp_rank
```

**Common cases:**

| Relationship | Token condition | Mapping |
| --- | --- | --- |
| `dcp_size_P == dcp_size_D` | `dcp_local_P == dcp_local_D` | 1 : 1 |
| `dcp_size_P > dcp_size_D` | `dcp_local_P % dcp_size_D == dcp_local_D` | N : 1 (P finer) |
| `dcp_size_P < dcp_size_D` | `dcp_local_D % dcp_size_P == dcp_local_P` | 1 : N (D finer) |
