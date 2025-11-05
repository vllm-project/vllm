# Logits Processors

!!! important
    Some logits processors design changes are still in progress and the API may
    change in the near future. We hope to stabilize this part of the API soon

This document describes how the vLLM engine interacts with logits processors, and the programming model which vLLM supports for implementing logits processors.

## Logits Processors Background

A logits processor adjusts the next-token probability distribution, usually with the intention of steering the model towards a desired type of behavior.

In vLLM, logits processors operate at batch granularity. During a given engine step, the logits processor consumes a `(num_requests) x (vocab_size)` tensor of raw logits output by the model. For all requests which enable the logits processor, the logits processor applies a transformation to the corresponding row of the logits tensor, while leaving other rows unmodified. The transformed logits tensor is then passed to softmax.  

## Logits Processors in the vLLM engine

The vLLM engine's persistent batch data structure maintains a list of loaded logits processors.

In order to operate on the entire batch at once, each logits processor may maintain metadata about the requests in the batch (i.e. each request's logits-processor-specific configuration settings). Therefore, logits processors are stateful.

In each engine step, the vLLM engine will (1) update each logits processor's internal state and (2) apply logits processors to the model output logits.

### Updating Logits Processor Internal State

At the beginning of each engine step, the persistent batch may add, discard and/or reorder requests in response to the scheduler output. After the persistent batch has reorganized, the vLLM engine invokes each logits processor's `update_state()` method. This is necessary to ensure that logits processors' internal states are reorganized to match the new persistent batch state at the beginning of the engine step.

The pseudocode below shows the process by which the vLLM persistent batch notifies each logits processor of changes in batch state:

??? code "Model Runner Updates Logits Processor States"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            self._update_states(scheduler_output)

            ...

        def _update_states(...):

            ...

            # ...update persistent batch to reflect new/finished requests & reordering
            # of requests within batch...

            ...

            self.input_batch.refresh_metadata()


    # gpu_input_batch.py

    class InputBatch:

        ...

        def refresh_metadata(self):

            ...

            # Update each logits processor's state to reflect persistent batch state
            batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
            for logit_proc in self.logitsprocs.all:
                logit_proc.update_state(batch_update)

            ...


    # vllm/v1/sample/logits_processor/interface.py

    @dataclass(frozen=True)
    class BatchUpdate:
        # Batch state-change data structure which is passed to logits processors'
        # update_state() methods

        batch_size: int

        removed: Sequence[RemovedRequest]
        added: Sequence[AddedRequest]
        moved: Sequence[MovedRequest]
    
    ```

### Applying Logits Processors to the Model Output Logits

After updating persistent batch state, the vLLM model runner performs model inference to obtain logits. Then, the model runner invokes the sampler against the logits. In turn, part of the sampler's operation is to invoke the logits processors' `apply()` methods against the model output logit processors, yielding transformed logits (the `apply()` methods may modify the logits in-place or out-of-place, although in-place is more memory-efficient). This process is shown in the pseudocode below.

Note that the sampler will access the logits processors via `SamplingMetadata.logitsprocs`. When the vLLM engine constructs `SamplingMetadata` (not shown in the code below), the reference to the list of logits processors is passed from the persistent batch data structure to `SamplingMetadata`.

??? code "Apply logits processors to model output logits"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            # (discussed in previous section)
            self._update_states(scheduler_output)

            ...

            # ...run model inference to obtain logits...

            ...

            # Invoke sampler, which applies logits processors
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=sampling_metadata)

            ...


    # sampler.py

    class Sampler(nn.Module):

        ...

        def forward(self, logits, sampling_metadata):

            ...

            # Apply non-argmax-invariant logits processors to model output logits
            for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
                logits = processor.apply(logits)

            sampled = self.sample(logits, sampling_metadata)

            ...

            # ...return sampler output data structure...


        def sample(self, logits, sampling_metadta)

            ...

            # ...exit early if all requests are greedy-sampling...

            ...

            # Apply argmax-invariant logits processors
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            ...

            # ...perform sampling and return sampling result...
    ``` 

At sampling time, the sampler checks whether all requests in the persistent batch employ greedy sampling. If that is the case, the sampler saves compute by skipping "argmax-invariant" logits processors. Here, "argmax" is shorthand for the token ID with the highest logit value in a given row of the logits tensor (i.e. the token which the model weighted the highest for a given request).

* An **argmax-invariant logits processor** is a logits processor (such as Min-P) which does not modify the argmax. For example, a logits processor which masks out the lowest-probability tokens will not change which token ID has the max logit. Greedy sampling always picks the highest-logit-value token ID, and so conceptually an argmax-invariant logits processor can be skipped for greedy sampling requests.

* A **non-argmax-invariant logits processor** is a logits processor which may modify the argmax. For example, a logits processor which masks all tokens except for EOS after a certain number of steps in order to force decoding to terminate might end up masking the max-logit-value token and therefore change the argmax. Conceptually, these logits processors cannot be skipped for greedy sampling requests.

The vLLM logits processor abstraction requires the engine to apply logits processors at batch granularity; therefore in practice the argmax-invariant logits processors can only be skipped when the entire batch uses greedy sampling.

## Logits Processor Programming Model

The previous sections alluded to the interfaces which vLLM logits processors must support. This section introduces in full the programming model for implementing logits processors that are compatible with the vLLM engine, including the `LogitsProcessor` base class and its interface methods as well as the `BatchUpdate` data structure for representing persistent batch state changes, both of which are shown in the code below:

??? code "`LogitsProcessor` base class and `BatchUpdate` data structure"

    ``` python
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from dataclasses import dataclass
    from enum import Enum, auto
    from typing import TYPE_CHECKING

    import torch

    from vllm import SamplingParams

    if TYPE_CHECKING:
        from vllm.config import VllmConfig


    class MoveDirectionality(Enum):
        # One-way i1->i2 req move within batch
        UNIDIRECTIONAL = auto()
        # Two-way i1<->i2 req swap within batch
        SWAP = auto()


    # (index, params, prompt_tok_ids, output_tok_ids) tuples for new
    # requests added to the batch.
    AddedRequest = tuple[int, SamplingParams, list[int], list[int]]

    # (index 1, index 2, directionality) tuples representing
    # one-way moves or two-way swaps of requests in batch
    MovedRequest = tuple[int, int, MoveDirectionality]

    # Batch indices of any removed requests.
    RemovedRequest = int


    @dataclass(frozen=True)
    class BatchUpdate:
        """Persistent batch state change info for logitsprocs"""
        batch_size: int  # Current num reqs in batch

        # Metadata for requests added to, removed from, and moved
        # within the persistent batch.
        #
        # Key assumption: the `output_tok_ids` list (which is an element of each
        # tuple in `added`) is a reference to the request's running output tokens
        # list; via this reference, the logits processors always see the latest
        # list of generated output tokens
        removed: Sequence[RemovedRequest]
        moved: Sequence[MovedRequest]
        added: Sequence[AddedRequest]


    class LogitsProcessor(ABC):

        @abstractmethod
        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool) -> None:
            raise NotImplementedError

        @abstractmethod
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @abstractmethod
        def is_argmax_invariant(self) -> bool:
            """True if logits processor has no impact on the
            argmax computation in greedy sampling.
            NOTE: may or may not have the same value for all
            instances of a given LogitsProcessor subclass,
            depending on subclass implementation.
            """
            raise NotImplementedError

        @abstractmethod
        def update_state(
            self,
            batch_update: "BatchUpdate" | None,
        ) -> None:
            """Called when there are new output tokens, prior
            to each forward pass.

            Args:
                batch_update is non-None iff there have been
                changes to the batch makeup.
            """
            raise NotImplementedError

        @classmethod
        def validate_params(cls, sampling_params: SamplingParams):
            """Validate sampling params for this logits processor.

            Raise ValueError for invalid ones.
            """
            return None

    ```

A vLLM logits processor must subclass `LogitsProcessor` and define (at minimum) the following methods:

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`: engine configuration data structure
    * `device`: hardware accelerator device info
    * `is_pin_memory`: flag indicating whether pin memory is available to support logits processor implementation

* `apply(self, logits: torch.Tensor) -> torch.Tensor`:
    * Consume a `(num_requests) x (vocab_size)` logits tensor (`logits`)
    * Apply logits processor transformation at batch granularity
    * Return a transformed `(num_requests) x (vocab_size)` logits tensor
    * You can modify the input logits processors in-place or out-of-place; in-place is more memory-efficient

* `is_argmax_invariant(self) -> bool`:
    * Return `True` if the logits processor is argmax invariant (never changes what is the highest-logit-value token ID for a given request), `False` if the logits processor may modify argmax
    * `is_argmax_invariant()` is evaluated once at startup; if `True`, vLLM will skip applying this logits processor in a given step when all requests use greedy sampling

* `update_state(self, batch_update: "BatchUpdate" | None) -> None`:
    * Consume a `BatchUpdate` data structure representing persistent batch state changes at the beginning of the current engine step
    * Use the `BatchUpdate` members to update logits processor internal state
    * **Note:** batch update data structure may be `None`, signaling no change to the batch constituents. In this case, the LogitsProcessor might still want to update its state based on the updated `output_token_ids` lists that it could have retained when they were added.

* `validate_params(cls, sampling_params: SamplingParams)`:
    * Raise `ValueError` if `SamplingParams` has invalid arguments (especially custom arguments) used by logits processor.
    * When request is sent to entrypoint, `validate_params()` will validate `SamplingParams` and refuse request with invalid arguments.

### `BatchUpdate` data structure

The `BatchUpdate` abstraction models the persistent batch as a list of requests, supporting the following operations to change batch state (note that the order in which the operations are mentioned below reflects the order in which they should be processed in `update_state()`):

* **Remove:** remove (without replacement) request at index `i`

    * A Remove is represented in `Batchupdate.removed` by an `int` (representing `i`)

    * Effect of remove-at-index on batch:

        ``` text
        Batch: [A,B,C]
        Remove @ i:  1

        =>

        New Batch: [A,x,C] # Discard B and leave an empty slot
        ```

* **Add:** add (or replace existing request with) a new request at index `i`. If a request is replaced, its associated state should be discarded.

    * An Add is represented in `Batchupdate.added` as a tuple of

        ``` text
        (index, new request SamplingParams, prompt token ids, output token ids)
        ```

    * `prompt token ids` and `output token ids` are references to the request's prompt token ids and output token ids lists, respectively. Note that the output token ids list grows with each engine step, and this growth is visible to the logits processor because output token ids are passed by reference. **This is important for LogitsProcessors that take into account the tokens generated so far**.

    * The implementation of the particular logits processor subclass determines whether or how the fields in the added request tuple are digested into an internal representation. For example, a logits processor that does not utilize prompt or output token ids may only need to utilize `index` and `SamplingParams` and discard the other tuple fields

    * If index `i` currently holds a request, a replacement occurs:

        ``` text
        Batch: [A,B,C]
        New request to be added @ i: D @ 1

        =>

        New Batch: [A,D,C] # Add D, discard B
        ```

    * If index `i` does not currently hold a request (because `i` is out of bounds of the current batch size):

        ``` text
        Batch: [A,B,C]
        New request to be added @ i: D @ 3

        =>

        New Batch: [A,B,C,D] # Add D, extending batch
        ```

* **Move:** move request at index `s` to index `d` OR swap requests at indices `s` and `d`

    * A Move is represented in `Batchupdate.moved` as a tuple of

        ``` text
        (s, d, UNIDIRECTIONAL or SWAP)
        ```

    * If the Move specifies `UNIDRECTIONAL`:

        * The request at index `s` is moved to index `d`; index `s` becomes an empty slot

            ``` text
            Batch: [A,x,C,D]
            Unidirectionally Move s -> d:  3 -> 1

            =>

            New Batch: [A,D,C,x] # Move D to 1, leaving empty slot at 3
            ```

        * If another request already resided at index `d`, it is replaced and discarded

            ``` text
            Batch: [A,B,C,D]
            Unidirectionally Move s -> d:  3 -> 1

            =>

            New Batch: [A,D,C,x] # Move D to 1, discarding B and leaving empty slot at 3
            ```

    * If the Move specifies `SWAP`, the requests at `s` and `d` exchange indices

        ``` text
        Batch: [A,B,C,D]
        Swap Move s <-> d:  3 <-> 1

        =>

        New Batch: [A,D,C,B] # Swap B and D
        ```

Additionally, the `BatchUpdate` data structure includes a representation (`batch_size`) of the size of the persistent batch at the beginning of the engine step.

### How the vLLM engine builds the `BatchUpdate` data structure

Logits processor `update_state()` implementations should assume the following model for how the model runner updates persistent batch state (expressed here in terms of the `BatchUpdate` abstraction):

1. Identify indices of requests which finished in the current engine step

2. Identify new requests introduced in the current step

3. Use Add operations to replace as many finished requests with new requests, in order of increasing index of the replaced request starting with the lowest index

4. Based on the relative number of new and finished requests:

    1. If the numbers of new and finished requests are the same, proceed to next step

    2. *If there are more new requests than finished requests:* apply Add operations to extend the batch with the remaining new requests which did not replace finished requests. Assign consecutive indices to these new requests, starting with `current_max_batch_index + 1`

    3. *If there are fewer new requests than finished requests:*

        * Apply Remove operations to finished requests which were not replaced with new requests. These removed request indices will necessarily be greater than the greatest index of the finished requests which were replaced in the previous step. The Removes may leave the batch in a non-contiguous state

        * **"Condense" the batch to be contiguous:** starting with the lowest-index empty slot (which was caused by a Remove), apply a Unidirectional Move from the current highest non-empty slot in the batch to fill the empty slot. Proceed with additional Unidirectional Move operations in order of increasing empty slot destination index and decreasing non-empty slot source index until the batch is contiguous

        * **Shrink the batch:** a side-effect of condensing the batch is that empty slots resulting from Remove operations are grouped in a contiguous block at the end of the batch array. Thus, after condensing, update `BatchUpdate.batch_size` to reflect the number of non-empty slots

5. Reorder the batch for improved efficiency. Depending on the attention backend implementation and the current characteristics of the batch, zero or more Swap Move operations may be applied to reorder the batch

Notes:

* A logits processor `update_state()` method must process batch update operations in the following order: removes, adds, moves

* The index argument for Add operations refers to the index *at the time the Add occurred*, i.e. before any Move operations
    * Example: if a request is Added at index 5 and then swapped with index 3, the Add operation in `BatchUpdate.added` will be associated with index 5 not 3
    * In other words Move operations can be assumed to be applied after Adds and Removes

* Move operations can be assumed to be applied in the order in which they appear in `BatchUpdate.moved`

* If there are no new/finished requests and there is no batch reordering, then the batch update for the logits processors will be `None`

#### Example: Batch Update with Fewer New Requests Than Finished Requests

The following example models an engine step where 1 new request is introduced and 2 finished requests are eliminated, additionally the attention backend performs a swap to optimize the batch ordering.

``` text
Batch state (beginning of engine step): [A,B,C,D]
Batch size: 4

New requests: E

Finished requests: A, C

Processing steps (using BatchUpdate abstraction):

1. Add E at index 0

[E,B,C,D] # Discard A
Batch size: 4

2. Remove at index 2

[E,B,x,D] # Discard C, empty slot at index 2
Batch size: 4

3. Condense batch with a Unidirectional Move 3 -> 2 operation and shrink batch

[E,B,D] x # Empty slot is now outside batch
Batch size: 3

4. Attention backend optimization: reorder batch with Swap 0 <-> 1

[B,E,D]
Batch size: 3

```

The resulting `BatchUpdate` data structure will look like

``` text
BatchUpdate instance
* added: [(0,E's SamplingParams,E's prompt tokens ref,E's output tokens ref)]
* removed: [2] # request C was removed without replacement
* moved: [(3,2,UNIDIRECTIONAL),(0,1,SWAP)]
```

#### Example: Batch Update with More New Requests Than Finished Requests

The following example models an engine step where 2 new requests are introduced and 1 finished request is eliminated, additionally the attention backend performs a swap to optimize the batch ordering.

``` text
Batch state (beginning of engine step): [A,B,C,D]
Batch size: 4

New requests: E,F

Finished requests: C

Processing steps (using BatchUpdate abstraction):

1. Add E at index 2

[A,B,E,D] # Discard C
Batch size: 4

2. Add F at index 4 (current max batch index + 1)

[A,B,E,D,F] # Extend batch by 1
Batch size: 5

4. Attention backend optimization: reorder batch with Swap 0 <-> 1

[B,A,E,D,F]
Batch size: 5

```

Note that batch condensation is skipped because there are no empty slots left behind by Remove operations.

The resulting `BatchUpdate` data structure will look like

``` text
BatchUpdate instance
* added: [(2,E's SamplingParams,E's prompt tokens ref,E's output tokens ref),(4,F's SamplingParams,F's prompt tokens ref,F's output tokens ref)]
* removed: [] # no requests were removed without replacement
* moved: [(0,1,SWAP)]
```

## How to Introduce a New Logits Processor to vLLM

### Best Practices for Writing Built-In Logits Processors

* Write efficient `apply()` and `update_state()` implementations in light of the fact that logits processors operate at batch granularity
    * For example, you may be able to use efficient vectorized operations to implement `apply()` or update internal state vectors in `update_state()`
    * However, if you think that a logits processor may be used infrequently, it may be appropriate to use a "sparse" representation of request state i.e. the class can represent request configuration using a dictionary which only stores metadata about requests that enable the logits processor

* It is up to the logits processor author to determine:

    1. **The per-request attributes which configure the logits processor's behavior against that request.** For example, if you are writing a new built-in logits processor for vLLM, you may or may not need to add additional fields to `SamplingParams` and the vLLM REST API

    2. **The conditions under which the logits processor is or is not enabled on a per-request basis.** Unless your intention is for the built-in logits processor to act on all requests all the time, you should write your logits processor in such a way that it is possible to disable the logits processor for a given request, i.e. by defaulting an argument to `None` or by passing in a specific do-nothing argument value i.e. `0.0`. Try to save compute and memory for requests which disable the logits processor

    3. **The conditions under which the logits processor is short-circuited at the batch level.** Even if you have defined a way to disable the built-in logits processor at the request level, it may be difficult to translate this into compute savings i.e. if your `update_state()` and `apply()` implementations use efficient vectorized implementations that operate on the whole persistent batch in a single command. For example, you cannot skip an entire vectorized operation in `apply()` just because one request disabled the logits processor. To save compute in the edge-case where no running requests utilize the built-in logits processor, we recommend designing `apply()` to return the unmodified input tensor if all requests have the logits processor disabled. Similarly, consider whether steps can be skipped in `update_state()` if no requests enable the logits processor

        * Additionally, an easy way to save compute in `update_state()` is to exit early when the batch_update is `None`

* Ensure that the logits processor `update_state` method discards information about finished requests (i.e. requests which are replaced by an Add or which are subject to a Remove)

* `is_argmax_invariant()` can be hard-coded to `True` or `False` if the logits processor has consistent behavior. However the argmax invariance may also be determined programmatically (i.e. if your logits processor is user-customizable in some way that impacts whether the logits processor is argmax invariant). For this reason, `is_argmax_invariant()` is not a class method

### Built-In Logits Processors

Built-in logits processors are always loaded when the vLLM engine starts. See the existing vLLM built-in logits processors in `vllm/v1/sample/logits_processor/builtin.py` for examples of how to write a new built-in vLLM logits processor. It makes sense to write a PR to introduce a new logits processor as a built-in if it is likely to be useful to a wide audience. vLLM currently employs the following built-in logits processors based on the programming model described above:

* Min-P

* Logit bias

* Min-tokens

Review these logits processor implementations for guidance on writing built-in logits processors.

Additionally, the following logits-processor-like functionalities are hard-coded into the sampler and do not yet utilize the programming model described above. Most of them will be refactored to use the aforemented logits processor programming model.

* Allowed token IDs

* Bad words

* Repetition penalty

* Frequency penalty

* Presence penalty

* Temperature

* Top-K

* Top-P

### Custom Logits Processors

vLLM can be augmented with [user-provided custom logits processors](../features/custom_logitsprocs.md).
