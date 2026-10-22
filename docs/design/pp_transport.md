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
| Hidden states, residual | Model inputs | Existing worker policy |
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
entry point delegates to `PPTransport`, so its callers share the same allocator
without changing their import paths. It selects the registered hidden-state and
residual fields used by existing models.

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
