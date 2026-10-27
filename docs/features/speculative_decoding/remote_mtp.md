# Standalone own-KV MTP

The experimental standalone runtime can place a target-conditioned MTP head in
another process. It uses the native MRV2 MTP forward path and transports target
hidden states. The target still verifies every proposed token. This placement
can reduce target GPU memory use; it does not imply faster decoding or overlap
between target and draft computation.

## Model contracts

The runtime checks the resolved draft architecture and its matching target,
including hidden width, vocabulary size and the number of checkpoint MTP modules.
It currently requires exactly one MTP module and scalar token positions.

| Resolved draft architecture | Matching target | Draft cache | K restriction |
| --- | --- | --- | --- |
| `MiMoMTPModel` | `MiMoForCausalLM` | Full attention | 1 |
| `ErnieMTPModel` | `Ernie4_5_MoeForCausalLM` | Full attention | 1 |
| `Qwen3NextMTP` | `Qwen3NextForCausalLM` | Full attention | Native sequential rollout |
| `Glm4MoeMTPModel` | `Glm4MoeForCausalLM` | Full attention | Native sequential rollout |
| `Glm4MoeLiteMTPModel` | `Glm4MoeLiteForCausalLM` | Dense MLA | Native sequential rollout |
| `ExaoneMoeMTP` | `ExaoneMoeForCausalLM` / `ExaoneMoEForCausalLM` | Uniform full attention or sliding window | Native sequential rollout |
| `NemotronHMTPModel` | `NemotronHForCausalLM` / `NemotronHPuzzleForCausalLM` | Attention/MoE-only MTP block | Native sequential rollout |
| `HYV3MTPModel` | `HYV3ForCausalLM` | Full attention | Native sequential rollout |

These are explicit implementation contracts, not validation of every checkpoint
or topology in each family. A target with hybrid attention or recurrent layers
can have a conventional attention-only MTP head. The target's architecture name
alone does not determine the draft's cache layout.

Exaone MTP modules use the draft checkpoint's module count, attention types and
sliding windows, including when the draft checkpoint is published separately
from the target.

Native unquantized HYV3 loading checks every MTP-owned parameter, including each
packed projection and local expert shard. Draft-only checkpoints may omit the
target-owned embedding and LM head; the native proposer installs them after
loading. Included tied heads load from the checkpoint embedding, and included
untied heads load from its LM head. An explicit native loader
`enable_weights_track=False` retains the existing unchecked-loading opt-out.

The standalone allocator accepts one homogeneous, unquantized full-attention,
dense MLA, or sliding-window group. It uses the shared vLLM KV allocator and
retains a null block. Sliding-window attention conservatively reserves the full
context; the attention backend applies the window, and this runtime does not
recycle evicted window pages. Budget the draft KV accordingly.

Sparse/indexer MLA (including any `index_topk` configuration), compressed or
packed KV, mixed KV groups, multiple MTP modules, multimodal features, mRoPE and
mHC are rejected. Nemotron MTP must contain only attention and MoE layers.
Gemma4's Q-only assistant reads target-owned KV and is not supported by this
own-KV adapter. Its native local path is unaffected.

## Shared weights and execution

The standalone process loads the target embedding and LM head without loading
its transformer layers. Automatic checkpoint downloads use the resolved target
revision. Nemotron's native `backbone.embeddings.weight` name is recognized;
ambiguous aliases, incompatible shapes and quantized embedding/head tensors are
rejected instead of being interpreted as ordinary floating-point weights.

Remote checkpoint identity uses a resolved immutable Hugging Face commit, or
SHA-256 digests of local checkpoint artifacts and shard indexes. Local directories
are identified by their contents rather than their deployment path. These files
are streamed once per contract construction at startup; hashing adds startup I/O,
not a per-round cost. Checkpoints must remain unchanged between contract creation
and model loading, and throughout serving; hot reload is unsupported. Local
tokenizer artifacts are likewise hashed independently of their directory path.

An alternate `--shared-weights-source` must be a local directory. Its selected
embedding and head tensors must match the target checkpoint in dtype, shape and
contents before GPU materialization. Unpinned alternate repositories and shape-
compatible replacement weights are rejected; this option stages identical shared
weights rather than changing the model being advertised by the handshake.

The runtime remains greedy, eager, same-host and draft TP1. Existing remote
restrictions on target topology, prefix caching, asynchronous scheduling,
resumable input, multimodal inputs and LoRA still apply. Native local MTP does
not inherit these placement restrictions. Set attention backends supported by
the actual draft layout: dense MLA requires an MLA backend; a full-attention
backend used by another checkpoint is not interchangeable.

Remote drafting also rejects KV consumer connectors and KV offloading, including
consumer children of a `MultiConnector`. External KV hits skip target prefill
even when prefix caching is disabled, leaving the remote drafter without the
context it requires. Producer-only connectors retain their existing validation.

## Validation boundaries

CPU tests cover real Transformers configuration normalization, architecture and
feature rejection, shared-weight selection, and the real KV allocator. The KV
tests check independent layer/block writes, dense MLA latent storage and kernel
block splitting, sliding-window rejection/cleanup, and exact budget boundaries.
These tests do not establish attention numerics or model output correctness.

Model validation must record the exact source and checkpoint revisions, compare
native and remote output, require nonzero proposed and accepted drafts, and
check cleanup. Same-GPU tests exercise process separation but do not measure
cross-GPU transport or memory savings. A checkpoint without recorded model
validation remains experimental even when its configuration passes the contract.
