# DeepSeek V4.1 Engram shared storage for sequence parallelism

DeepSeek V4.1 normally partitions Engram tables by hash head. Each tensor-parallel
rank looks up its heads for all tokens, gathers its peers' embeddings, and selects
its sequence-parallel token slice. The opt-in `sp_shared_memory` mode instead
makes the complete CPU table visible to every TP rank. Each rank looks up all
heads for its own tokens, producing the projection's input without embedding
all-gather or token selection.

```text
Default
head-sharded table -> local heads for all tokens -> TP all-gather
                  -> select local tokens -> projection and gate

sp_shared_memory
full shared table -> all heads for local tokens -> projection and gate
```

## Enable

On a supported same-host CUDA configuration with sequence parallelism enabled:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
    --tokenizer-mode deepseek_v41 \
    --language-model-only \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}' \
    --engram-config '{"cpu_offload":true,"sp_shared_memory":true}'
```

The option defaults to `false`. Initial support requires:

- DeepSeek V4.1, CUDA, CPU offload and model runner V2.
- TP greater than one with every TP rank on the same host and sharing an IPC
  namespace; DP1 and PP1.
- Active sequence parallelism. TP alone does not enable it: this DP1 model needs
  expert parallelism and a mega-MoE backend. The example was qualified on GB200.
- Single-threaded `auto`, `safetensors` or `pt` loading.
- No context parallelism, microbatching/DBO, speculative decoding, elastic EP,
  sleep mode, `dp_shared_memory` or `embedding_across_dp`.

Unsupported enabled configurations fail before table allocation where possible.
Disabling the option restores the ordinary head-sharded implementation.

## Storage and loading

The implementation reuses registered shared host storage. Each Engram table is
backed by one temporary `/dev/shm` mapping, registered with CUDA by each TP
process. The file is unlinked after the ranks have mapped it; live tensor aliases
retain the allocation. Sufficient shared-memory capacity and host RAM are needed.

Checkpoint parameters retain their original TP head-shard shapes and offsets.
Each rank writes only its assigned rows into the common backing. A CPU-group
collective publishes completion or a loading error before execution proceeds.
Replacing a registered parameter's storage is rejected.

For this checkpoint, the two FP8 tables and their scale bytes occupy about
188.83 GiB in total. Mapping that full address range in four processes does not
create four physical copies. It also does not reduce table memory relative to
one ordinary TP-sharded replica. Registration costs and NUMA placement still
matter: full visibility does not guarantee local host access for every GPU.

## Forward execution

N-gram hashes are computed with complete request history before SP slicing.
Only the resulting hash IDs are sliced. The existing GPU UVA lookup reads host
FP8 rows and UE8M0 scales, dequantizes them, and writes all heads for the local
SP tokens into persistent BF16 HBM. The kernel writes zeros for padded rows,
including ranks whose token slice is empty.

WKV, activation quantization and gated residual injection are unchanged. Lookup
prefetch retains the existing stream policy. At TP4 with an 8192-token ceiling,
staging changes from `[8192, 6, 256]` to `[2048, 24, 256]`: both reserve 24 MiB
per layer per rank. No full HBM copy of the host tables is introduced.

This removes Engram's TP exchange and selection copy, not other model
collectives or host reads. Throughput depends on workload and host topology;
communication elimination is not a general speedup guarantee.
