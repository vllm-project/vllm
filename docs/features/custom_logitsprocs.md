# Custom Logits Processors

This document shows you how to augment vLLM with custom logits processors.

## Build a custom logits processor and pass it to the offline `LLM` engine

Subclass `vllm.v1.sample.logits_processor.LogitsProcessor` and override the following methods
* `__init__(self, vllm_config: vllm.config.VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`: vLLM engine configuration
* `is_argmax_invariant(self)`
* `update_state(self, batch_update: Optional[vllm.v1.sample.logits_processor.BatchUpdate])`
    * `batch_update`: representation of added/removed/moved requests in the vLLM persistent batch during the most recent engine step
* `apply(self, logits: torch.Tensor)`
    * `logits`: a $num\_reqs \times vocab\_size$ tensor representing the unprocessed token probability distribution for each request. 

The contrived example below implements a 

??? code "Example custom logits processor definition"

    ```python
    from typing import Optional
    import torch
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import (BatchUpdate,
                                                LogitsProcessor,
                                                MoveDirectionality)

    class DummyLogitsProcessor(LogitsProcessor):
        """Fake logit processor to support unit testing and examples"""

        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool):
            self.req_info: dict[int, SamplingParams] = {}

        def is_argmax_invariant(self) -> bool:
            """Never impacts greedy sampling"""
            return False

        def update_state(self, batch_update: Optional[BatchUpdate]):
            if not batch_update:
                return

            # Process added requests.
            for index, params, _, _ in batch_update.added:
                assert params is not None
                if params.extra_args and (target_token :=
                                        params.extra_args.get("target_token")):
                    self.req_info[index] = target_token

            if self.req_info:
                # Process removed requests.
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # Process moved requests, unidirectional move (a->b) and swap
                # (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            # Save target values before modification
            rows_list = list(self.req_info.keys())
            cols = torch.tensor([self.req_info[i] for i in rows_list],
                                dtype=torch.long,
                                device=logits.device)
            rows = torch.tensor(rows_list, dtype=torch.long, device=logits.device)
            values_to_keep = logits[rows, cols].clone()
    ```

Pass your custom logits processor to the `LLM` constructor in the form of (1) a class object or (2) a fully-qualified class name (FQCN), as shown in the example below (which assumes that `DummyLogitsProcessor` is defined in `vllm.test_utils`):

```
# Pass in class object
llm = LLM(
    model="facebook/opt-125m",
    logits_processors=[DummyLogitsProcessor],
)

# Pass in FQCN
llm = LLM(
    model="facebook/opt-125m",
    logits_processors=["vllm.test_utils:DummyLogitsProcessor"],
)
```

## Online scenario: pass the logits processor FQCN via CLI with `--logits-processors`

??? console "Launch vLLM OpenAI API-compatible server with custom logits processor"

    ```bash
    $ vllm serve facebook/opt-125m --logits-processors vllm.test_utils:DummyLogitsProcessor
    ```