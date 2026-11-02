# Pipeline Side-Input Transport

`vllm.distributed.pp_transport` describes tensors carried by the existing
pipeline-parallel tensor-dict channel. It does not add collectives or change
sampled-token synchronization. Each PP stage selects its transport fields at
initialization; the communication policy is shared across stages.

## Registering Data Types

Add a field to `PPTransportDataType` and `PP_TRANSPORT_REGISTRY` before using it.
The registry defines its wire name, whether it belongs in the model's compiled
input schema, and its TP all-gather policy. Indexed payloads can carry a list of
tensors, and companions select related fields together.

| Field | Location | TP policy |
| --- | --- | --- |
| Hidden states, residual | Model inputs | Existing worker policy, unless SP-sharded |
| Shared top-k indices | Model inputs | Matching TP lane |
| Multimodal embeddings and CPU mask | Graph-external payload | Matching TP lane |

Unknown `pp_transport.*` fields are rejected. Other existing intermediate tensor
keys retain their original behavior. Upstream auxiliary-hidden-state transport
remains owned by `PPHandler`.

## Model Initialization

The model selects only the fields it needs. Shapes exclude the leading token
dimension; a bound buffer supplies its shape and dtype. Without an explicit
dtype, the empty-tensor factory uses the runner's requested dtype.

```python
self.pp_transport = PPTransport({
    PPTransportDataType.HIDDEN_STATES: PPTransportTensor(shape=(hidden_size,)),
    PPTransportDataType.RESIDUAL: PPTransportTensor(shape=(hidden_size,)),
    PPTransportDataType.TOPK_INDICES: PPTransportTensor(
        buffer=self.topk_indices_buffer,
        receive=receive_previous_topk,
        send=send_next_topk,
    ),
})
self.make_empty_intermediate_tensors = (
    self.pp_transport.make_empty_intermediate_tensors
)
```

Expose `pp_transport` on the top-level model too (for example,
`self.pp_transport = self.model.pp_transport`). MRV2's worker and graph manager
read this binding; models without one retain the full-tensor layout.

Receive and send selections are independent. For example, a stage can receive
the preceding layer's top-k but omit it from its output when the next stage
computes a fresh index. The receiving stage's empty-tensor schema includes only
its selected incoming fields.

In the model's forward, restore persistent buffers before running layers and
package the outgoing tensors afterward:

```python
self.pp_transport.restore_buffers(intermediate_tensors)
# Run this stage's layers.
return self.pp_transport.make_intermediate_tensors(
    num_tokens, hidden_states=hidden_states, residual=residual
)
```

Restoring copies active rows into the original buffer, preserving addresses used
by attention layers and graph capture. Sending slices the buffer to the active
token count; the caller must retain it until the existing send completes.

## Migrating Existing Models

The existing `model_executor.models.utils.make_empty_intermediate_tensors_factory`
entry point shares the allocator without changing its import path, arbitrary
legacy field names, or optional `buffers` argument. The DeepSeek V3.2 NVIDIA
backbone uses the typed schema for hidden states and shared top-k indices,
retaining the existing layer-sharing boundary detection.

## Per-Tensor Sequence-Parallel Layouts

A complete, TP-replicated tensor uses the existing worker optimization: each TP
lane sends one slice, then the receiver all-gathers the slices. An SP tensor is
already a local shard, so it must be sent directly to the matching TP lane,
without slicing or all-gathering it again. Declare this per field, not per model:

```python
self.pp_transport = PPTransport(
    {
        PPTransportDataType.HIDDEN_STATES: PPTransportTensor(
            shape=(hidden_size,), sequence_parallel=True,
        ),
        PPTransportDataType.RESIDUAL: PPTransportTensor(shape=(hidden_size,)),
        PPTransportDataType.TOPK_INDICES: PPTransportTensor(
            buffer=self.topk_indices_buffer,
        ),
    },
    tp_size=get_tp_group().world_size,
)
```

For `N` global (runner-padded) tokens and `T` TP ranks, SP fields allocate and
slice `ceil(N / T)` rows; full fields keep `N` rows. The model must produce this
padded shard layout, and adjacent stages must agree on each field's layout and
TP rank ordering. The schema does not perform a model's SP collectives or
padding itself. `make_intermediate_tensors` receives the global token count,
not the shard's row count.

MRV2 uses `prepare_inputs` for dummy runs, graph capture, and runtime receive
copies. Persistent input buffers retain their addresses. Full-graph output
copies and replay views preserve each field's captured shape, so a batch may
contain both local shards and full tensors. Graph-external payloads are unchanged.

SP bindings are opt-in for MRV2. This framework does not enable SP+PP on models
that currently disallow it, and does not alter MRV1's communication policy.
SP-sharded PP inputs with ubatching/DBO are rejected: its global-token slicing
would require a separate shard-aware implementation. Existing full-tensor PP
and ubatching retain their behavior.

## Graph-External Payloads

Select runtime side inputs separately from the model input schema:

```python
transport = PPTransport(
    payload_types=(PPTransportDataType.MULTIMODAL_EMBEDDINGS,)
)
payload = transport.new_payload()
payload.set_multimodal_embeddings(embeddings, is_multimodal)
outgoing = payload.relay(output_intermediate_tensors)
```

Receiving stages reconstruct a fresh payload with
`transport.new_payload(received_tensors)`. This keeps batches independent and
does not add variable-length embedding lists to compiled model inputs.
Selecting multimodal embeddings also selects their CPU mask. Modality labels
are preserved in the wire keys; the mask stays on CPU without a device-to-host
copy. All relaying stages must select the same payload types.

MRV2 enables this relay when the loaded draft supports multimodal inputs. Only
multimodal encoder outputs and their mask are relayed; text embedding activations
and embedding weights are not transported. Drafters retain their local text
embedding lookup.
