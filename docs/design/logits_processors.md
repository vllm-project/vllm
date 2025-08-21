# Logits Processor Support in vLLM

This document describes how the vLLM engine interacts with logits processors, and the programming model which vLLM supports for implementing logits processors.

## Logits Processors Background

A logits processor adjusts the next-token probability distribution, usually with the intention of steering the model towards a desired type of behavior.

In vLLM, logits processors operate at batch granularity. During a given engine step, the logits processor consumes a `(num_requests) x (vocab_size)` tensor of raw logits output by the model. For all requests which enable the logits processor, the logits processor applies a transformation to the corresponding row of the logits tensor, while leaving other rows unmodified. The transformed logits tensor is then passed to softmax.  

## Logits Processors in the vLLM engine

The vLLM engine's persistent batch data structure maintains a list of loaded logits processors.

In order to operate on the entire batch at once, each logits processor may maintain metadata about the requests in the batch (i.e. each request's logits-processor-specific configuration settings). Therefore, logits processors are stateful.

In each engine step, the vLLM engine will (1) update each logits processor's internal state and (2) apply logits processors to the model output logits.

### Updating logits processor internal state

The vLLM model runner invokes each logits processor's `update_state()` method at the end of each engine step. This is necessary to ensure that logits processors' internal states are reorganized to match the new persistent batch state at the end of the current step. The pseudocode below shows that the vLLM model runner computes updates to the persistent batch state and then notifies each logits processor of the state changes:

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

            # Update persistent batch to reflect new/finished requests & reordering
            # of requests within batch

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


    # logits_processor/interface.py

    @dataclass(frozen=True)
    class BatchUpdate:
        # Batch state-change data structure which is passed to logits processor
        # update_state() method

        batch_size: int

        removed: Sequence[RemovedRequest]
        moved: Sequence[MovedRequest]
        added: Sequence[AddedRequest]
    
    ```

    !!! note
        `InputBatch.refresh_metadata()` generates a `BatchUpdate` data structure - representing the persistent batch state changes resulting from new, finished and reordered requests - and passes that data structure to the logits processors' `update_state()` methods.

### Applying logits processors to the model output logits

The pseudocode below shows how the vLLM model runner invokes the sampler, which in turn invokes the logits processors' `apply()` methods against the model output logit processors.

Note that the sampler will access the logits processors via `SamplingMetadata.logitsprocs`. When the vLLM engine constructs `SamplingMetadata`, the reference to the list of logits processors is passed from the persistent batch data structure to `SamplingMetadata`.

??? code "Apply logits processors to model output logits"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):

            ...

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

            # Return sampler output data structure


        def sample(self, logits, sampling_metadta)

            ...

            # Exit early if all requests are greedy-sampling

            ...

            # Apply argmax-invariant logits processors
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            ...

            # Perform sampling and return sampling result
    ``` 

At sampling time, the engine saves compute by skipping "argmax-invariant" logits processors in the edge-case where all requests employ greedy sampling. Here, "argmax" is shorthand for the token ID with the highest logit value in a given row of the logits tensor (i.e. the token which the model weighted the highest for a given request).

* An **argmax-invariant logits processor** is a logits processor (such as Min-P) which does not modify the argmax. For example, a logits processor which masks out the lowest-probability tokens will not change which token ID has the max logit. Greedy sampling always picks the highest-logit-value token ID, and so conceptually an argmax-invariant logits processor can be skipped for greedy sampling requests.

* A **non-argmax-invariant logits processor** is a logits processor which may modify the argmax. For example, a logits processor which masks all tokens except for EOS after a certain number of steps in order to force decoding to terminate might end up masking the max-logit-value token and therefore change the argmax. Conceptually, these logits processors cannot be skipped for greedy sampling requests.

The vLLM logits processor abstraction requires the engine to pass in state updates at batch granularity; therefore in practice state updates for argmax-invariant logits processors can only be skipped when the entire batch uses greedy sampling.

## Logits Processor Programming Model

The previous sections alluded to the interfaces which vLLM logits processors must support. This section introduces in full the programming model for implementing logits processors that are compatible with the vLLM engine, including the `LogitsProcessor` base class and its interface methods as well as the `BatchUpdate` data structure for representing persistent batch state changes, both of which are shown in the code below:

??? code "`LogitsProcessor` base class and `BatchUpdate` data structure"

    ``` python
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from dataclasses import dataclass
    from enum import Enum, auto
    from typing import TYPE_CHECKING, Optional

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
            batch_update: Optional["BatchUpdate"],
        ) -> None:
            """Called when there are new output tokens, prior
            to each forward pass.

            Args:
                batch_update is non-None iff there have been
                changes to the batch makeup.
            """
            raise NotImplementedError
            
    ```

A vLLM logits processor must subclass `LogitsProcessor` and define (at minimum) the following methods:

* `__init__()`

* `apply(self, logits: torch.Tensor) -> torch.Tensor`:
    * Consume a `(num_requests) x (vocab_size)` logits tensor (`logits`)
    * Apply logits processor transformation at batch granularity
    * Return a transformed `(num_requests) x (vocab_size)` logits tensor

* `is_argmax_invariant(self) -> bool`:
    * Return `True` if the logits processor is argmax invariant (never changes what is the highest-logit-value token ID for a given request), `False` if the logits processor may modify argmax
    * `is_argmax_invariant()` is evaluated once at startup; if `True`, vLLM will skip applying this logits processor in a given step when all requests use greedy sampling

* `update_state(self, batch_update: Optional["BatchUpdate"]) -> None`:
    * Consume a `BatchUpdate` data structure representing persistent batch state changes at the end of the current engine step
    * Batch update data structure may be `None`, signaling no state-change

### `BatchUpdate` data structure

The `BatchUpdate` abstraction models the persistent batch as a list of requests, supporting the following operations to change batch state (summarized below along with a schematic representation of how the batch is modified by the operation):

* **Add:** add (or replace existing request with) a new request at index `i`

    * An Add is represented in `Batchupdate.added` as a tuple of

        ``` text
        (index, new request SamplingParams, prompt token ids, output token ids)
        ```

    * `prompt token ids` and `output token ids` are references to the request's prompt token ids and output token ids lists, respectively. Note that the output token ids list grows with each engine step, and this growth is visible to the logits processor because output token ids are passed by reference

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

* **Remove:** remove (without replacement) request at index `i`

    * A Remove is represented in `Batchupdate.removed` by an `int` (representing `i`)

    * Effect of remove-at-index on batch:

        ``` text
        Batch: [A,B,C]
        Remove @ i:  1

        =>

        New Batch: [A,x,C] # Discard B and leave an empty slot
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

Additionally, the `BatchUpdate` data structure includes a representation (`batch_size`) of the size of the persistent batch at the end of the engine step.

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

* The index argument for Add and Remove operations refers to the index *at the time the Add or Remove occurred*, i.e. before any Move operations
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

2. Add E at index 4 (current max batch index + 1)

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

    2. **The conditions under which the logits processor is or is not enabled on a per-request basis.** Unless your intention is for the custom logits processor to act on all requests all the time, you should write your logits processor in such a way that it is possible to disable the logits processor for a given request, i.e. by defaulting an argument to `None` or by passing in a specific do-nothing argument value i.e. `0.0`. Try to save compute and memory for requests which disable the logits processor

    3. **The conditions under which the logits processor is short-circuited at the batch level.** Even if you have defined a way to disable the custom logits processor at the request level, it may be difficult to translate this into compute savings i.e. if your `update_state()` and `apply()` implementations use efficient vectorized implementations that operate on the whole persistent batch in a single command. For example, you cannot skip an entire vectorized operation in `apply()` just because one request disabled the logits processor. To save compute in the edge-case where no running requests utilize the custom logits processor, we recommend designing `apply()` to return the unmodified input tensor if all requests have the logits processor disabled. Similarly, consider whether steps can be skipped in `update_state()` if no requests enable the logits processor

        * Additionally, an easy way to save compute in `update_state()` is to exit early when the batch_update is `None`

* Ensure that the logits processor `update_state` method discards information about finished requests (i.e. requests which are replaced by an Add or which are subject to a Remove)

* `is_argmax_invariant()` can be hard-coded to `True` or `False` if the logits processor has consistent behavior. However the argmax invariance may also be determined programmatically (i.e. if your logits processor is user-customizable in some way that impacts whether the logits processor is argmax invariant). For this reason, `is_argmax_invariant()` is not a class method

### Built-In Logits Processors

Built-in logits processors are always loaded when the vLLM engine starts. See the existing vLLM built-in logits processors in `logits_processor/builtin.py` for examples of how to write a new built-in vLLM logits processor. It makes sense to write a PR to introduce a new logits processor as a built-in if it is likely to be useful to a wide audience. vLLM currently supports the following built-in logits processors based on the programming model described above:

* Min-P

* Logit bias

* Min-tokens

Review these logits processor implementations for guidance on writing built-in logits processors.

Additionally, the following logits processors or logits-processor-like functionalities are hard-coded into the sampler for efficiency and do not utilize the programming model described above, but may be updated to use the aforemented logits processor programming model in the future:

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
