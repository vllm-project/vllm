# Logits Processors Programming Model

This document describes how the vLLM engine interacts with logits processors, and the programming model which vLLM supports for implementing logits processors.

## Logits Processors Background

A logits processor adjusts the next-token probability distribution, usually with the intention of steering the model towards a desired type of behavior. 

In vLLM, logits processors operate at batch granularity: during a given engine step, the logits processor consumes a $(num_requests) \times (vocab_size)$ tensor of raw logits output by the model. For all requests which enable the logits processor, the logits processor applies a transformation to the corresponding row of the logits tensor, while leaving other rows unmodified. The transformed logits tensor is then passed to softmax.  

## Logits Processors in the vLLM engine

The vLLM engine's persistent batch data structure maintains a list of loaded logits processors. This list is passed to `SamplingMetadata` when the data structure is built.

In order to operate on the entire batch at once, each logits processor may maintain metadata about the requests in the batch, such as whether each requests enables the logits processor as well as each request's configuration settings. Therefore, logits processors are stateful.

In each engine step, the vLLM engine will:

1. **Update each logits processor's internal state to match persistent batch internal state, by invoking each logits processor's `update_state()` method.** this is necessary to ensure that logits transformations are applied to the correct requests with the correct configuration settings; to ensure that the logits processors discard information about finished requests; and to allow certain logits processors to count decoding steps (example: limiting the max number of generated tokens requires counting the number of generated tokens for each request which uses the logits processor.) The pseudocode below shows how the vLLM model runner computes updates to the persistent batch state and then notifies each logits processor of the state changes:

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

2. **Apply the logits processors to the model output logits tensor, by invoking each logits processor's `apply()` method.** The pseudocode below shows how the vLLM model runner invokes the sampler, which in turn causes the logits processors to transform the model output logits.

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

### Updating logits processor state to match persistent batch state