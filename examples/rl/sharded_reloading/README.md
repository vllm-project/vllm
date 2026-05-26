# M-to-N weight transfer using Etha

> **Status: experimental, work in progress.** This example demonstrates
> M-to-N sharded weight reloading for RL post-training. APIs, file
> layout, handler taxonomy, and topology assumptions are all subject to
> change.
>
> Based on [cmriat/Etha](https://github.com/cmriat/Etha). The planner
> in `etha_sharding.py` is a vendored / adapted version of Etha's
> `comm/` module; the rest of this directory is the vLLM-side wiring.

The standard `rlhf_nccl_fsdp_ep.py` flow ships weights by gathering the
trainer's sharded parameters to rank 0 and broadcasting the full
tensors to every vLLM worker. That works but wastes bandwidth: every
inference rank receives the full model even though it only keeps the
slice that matches its own (DP, TP, EP) shard.

Etha avoids the gather + broadcast entirely. Both sides declare their
sharding as `(DeviceMesh, Placement)`. A planner figures out — at
init time — which source-rank slice maps onto which destination-rank
slice, and on every sync round bytes move straight from each source
shard into the corresponding destination shard over NCCL point-to-point.
For an M-trainer / N-inference setup, this is "M-to-N" weight transfer.

## Files

- [etha_chunk.py](etha_chunk.py) — the `Chunk` dataclass: one planned
  send / recv / self-copy on a tensor slice. Transport-agnostic
  (knows nothing about NCCL). `map_to_chunk_ops` specializes an
  abstract chunk-index-space M2M map into concrete per-rank Chunks.
- [etha_sharding.py](etha_sharding.py) — the planner. Declares the
  per-handler placement tables (`TRAINER_HANDLER_PLACEMENTS`,
  `VLLM_HANDLER_PLACEMENTS`), implements the trace-based
  mark-and-recapture M2M planner (`get_m2m_map`), and provides two
  concrete strategies:
    - `VllmEthaShardingStrategy` (role = `"tgt"`, receives) discovers
      its placements by walking the loaded vLLM model and
      cross-checks against the static table so placement drift fails
      loudly at startup.
    - `TrainerEthaShardingStrategy` (role = `"src"`, sends) reads
      placements directly from the static tables.
- [etha_engine.py](etha_engine.py) — wiring into vLLM's
  `WeightTransferEngine` ABC. Owns the init / update info
  dataclasses, the NCCL transport (`chunk_comm`), the worker-side
  engine (`EthaWeightTransferEngine`), the trainer-side engine
  (`EthaTrainerWeightTransferEngine`), and the backend registration
  hook (`EthaWorkerExtension`).
- [rlhf_etha.py](rlhf_etha.py) — runnable end-to-end example. Same
  shape as `rlhf_nccl_fsdp_ep.py` (load HF weights into a sharded
  DTensor state dict on 4 trainer Ray actors, start an
  `AsyncLLMEngine` with `load_format="dummy"`, generate gibberish,
  ship weights, generate coherent text), but with Etha instead of
  rank-0 gather + broadcast.

## How it works

### Planning phase (once, at init)

1. Trainer ranks and vLLM workers all join a single
   `StatelessProcessGroup` of size M + N.
2. Each side builds a "pair table" — one entry per handler
   (`qkv_proj`, `o_proj`, `embed_tokens`, `lm_head`, `router`,
   `layernorm`, `experts_gate_up`, `experts_down`) containing its
   own and the peer's `(DeviceMesh, Placement)`.
3. For each pair, `get_m2m_map` runs a mark-and-recapture trace:
   each source rank fills its local shard of a small LCM-sized
   "middle tensor" with an encoded `(rank, local_idx)` fingerprint,
   DTensor `full_tensor()` reassembles it, the assembled tensor is
   shipped (gloo, CPU) to the target ranks, and `distribute_tensor`
   reshards by the target placements. Reading the cells decodes the
   per-chunk `(src_rank, src_idx) → (dst_rank, dst_idx)` map.
4. `map_to_chunk_ops` specializes that abstract map into concrete
   `Chunk`s for this rank — each one is "send this slice to that
   rank" or "recv that slice from this rank".

The plan is cached on the engine; subsequent sync rounds never replan.

### Transfer phase (each sync round)

`chunk_comm` walks the pre-baked chunk list, prepares contiguous
send / recv buffers (downcasting to `wire_dtype` on the send side
where needed), opens one NCCL group, issues every `send` / `recv`,
closes the group, syncs the stream, and copies any dtype-converted
intermediates back into their destination slices. Self-copies (rare:
only when source and target meshes overlap on a rank) happen outside
the NCCL group as host-side `Tensor.copy_`.

Because chunks reference tensor views, the trainer mutates its
parameters in place between rounds — no re-registration needed.

## Topology (hardcoded today)

The example is hard-coded for **Qwen3-30B-A3B-Instruct-2507** on
8 GPUs single-node:

- Trainer: 4 Ray actors, `att_mesh = (dp_replicate=2, dp_shard=2)`,
  `moe_mesh = (dp_replicate=1, dp_shard=2, ep=2)`.
- Inference: 4 vLLM workers, `DP=2`, `TP=2`, `EP_SIZE = DP * TP = 4`.

The handler taxonomy (`HANDLER_KEYWORDS`, `VLLM_HANDLER_PLACEMENTS`,
`TRAINER_HANDLER_PLACEMENTS`) is Qwen3-MoE-specific. Lifting it into
a model-config-driven thing is on the TODO list.

## Running the example

From a vLLM environment with Ray and `transformers` available:

```bash
python examples/rl/sharded_reloading/rlhf_etha.py
```

Expected output:

1. Trainer actors load Qwen3-MoE weights via `dcp.load` straight onto
   sharded DTensors (no full-precision CPU staging).
2. vLLM generates from prompts using dummy weights — gibberish.
3. Etha rendezvous + plan trace (logs `Etha planner ... took ...s for
   N pairs`).
4. One weight transfer over NCCL P2P (logs `chunk_comm ... N ops,
   X GiB on the wire ...`).
5. vLLM generates again — coherent Qwen3 output.

## Known limitations / MVP shortcuts

- **Qwen3-MoE only.** Handler taxonomy and placement tables are hard-coded
  for that model.
- **In-place writes; layerwise reload would clobber.** The engine
  writes straight into the kernel-format parameter storage.
- **No online quantization.** Because chunks write directly into the
  kernel-format storage that `process_weights_after_loading` produced
  at startup, there is no opportunity to re-run a quantization pass
  per sync round. Trainer and inference dtype must match the storage
  dtype (modulo the `wire_dtype` downcast `chunk_comm` does on the
  send side). Mixing in a quantized inference backend that expects
  per-update re-quantization is not supported today.
- **Gloo + mesh caching workaround.** The planner shares one cached
  `DeviceMesh` per unique mesh tensor to dodge a gloo deadlock observed
  when multiple live PGs share the same rank composition. See the
  long comment in `EthaShardingStrategy.plan_and_specialize` if you
  hit a hang in `distribute_tensor` and need to debug.

## Credits

The trace-based M2M planner, the `(DeviceMesh, Placement)` per-handler
declarative style, and the overall approach are from
[cmriat/Etha](https://github.com/cmriat/Etha). This directory adapts
that work to vLLM's `WeightTransferEngine` interface and the
`AsyncLLMEngine` pause / update / resume flow.
