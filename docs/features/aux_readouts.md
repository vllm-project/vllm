# Auxiliary Readouts

A model can declare *auxiliary readouts*: small float32 outputs, such as an
N-way decision head, computed from the same hidden states that generation
samples from. A request asks for them by name, and they come back with its
completion. Other requests in the same batch are not affected, and no readout
work is done when no scheduled request asks. Model Runner V2 only.

## Model side

Implement `SupportsAuxReadout`:

```python
class MyModelWithHead(MyModelForCausalLM):
    supports_aux_readout = True
    aux_readout_dims = {"decision": 3}

    def compute_aux_readouts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [num_rows, hidden_size], the rows compute_logits reads
        return self.decision_head(hidden_states).float()  # [num_rows, 3]
```

`compute_aux_readouts` returns all declared readouts concatenated in the order
of `aux_readout_dims`, as one `[num_rows, sum(aux_readout_dims.values())]`
tensor; any other shape is an error. During chunked prefill, the row of a chunk
that does not finish the prompt is computed and discarded, like its logits.

## Request side

```python
from vllm import LLM, SamplingParams

llm = LLM("my-org/my-model-with-head", enable_aux_readouts=True)
params = SamplingParams(max_tokens=1, aux_readouts=["decision"])
completion = llm.generate("...", params)[0].outputs[0]
completion.aux_readouts["decision"]  # float32 tensor [len(token_ids), 3]
```

- One row is returned per generated token, at the row its logits were computed
  from. `max_tokens=1` gives a single decision.
- Requests without `aux_readouts` get `None`.
- Without `--enable-aux-readouts`, requests that set `aux_readouts` are
  rejected and the model runner keeps no readout state.
- Readouts are returned with the final output only, so
  `output_kind=RequestOutputKind.DELTA` is rejected.
- Not supported yet: speculative decoding, tensor, pipeline and context
  parallelism, and the OpenAI-compatible HTTP responses.
