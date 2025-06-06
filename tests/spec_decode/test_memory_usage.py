# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This docstring details important information on the testing methodology.

This test verifies that memory usage remains constant (or never grows) when 
we enable / disable speculation via --speculative-disable-by-batch-size. 

There are a lot of things we try to keep track of between batches of requests
and if certain tensors are not freed from memory, can result in CUDA ooms. 

This is particularly relevant for production situations where speculation might 
be enabled during off hours, but disabled once traffic peaks during the workday.
Since traffic will stay high for a long period of time, verifying we do not 
increase our memory usage over time is essential to prevent possible CUDA ooms. 
"""

import torch

import vllm
from tests.core.utils import create_dummy_prompt
from vllm.sequence import SequenceGroup

ITERATIONS = 100
MAIN_MODEL = "JackFram/llama-68m"

# speculative model
SPEC_MODEL = "abhigoyal/vllm-medusa-llama-68m-random"

BATCH_SIZE = 5
SPEC_DISABLE_BATCH_SIZE = 2


def add_seq_group_to_engine(engine: vllm.LLMEngine, seq_group: SequenceGroup):
    scheduler = engine.scheduler[0]
    scheduler.add_seq_group(seq_group)


"""
Since we are using a batch size greater than the disabled batch size, 
we can ensure we go through the _no_spec codepath for most of our engine steps.
"""


def test_memory_usage_no_spec():
    previous_memory_allocated = None
    llm = vllm.LLM(model=MAIN_MODEL,
                   speculative_config={
                       "model": SPEC_MODEL,
                       "num_speculative_tokens": 3,
                       "disable_by_batch_size": SPEC_DISABLE_BATCH_SIZE,
                   })

    batch_sequences = set()
    engine = llm.llm_engine

    for i in range(ITERATIONS):
        seq, seq_group = create_dummy_prompt(request_id=str(i),
                                             prompt_length=10,
                                             min_tokens=10,
                                             max_tokens=10)

        add_seq_group_to_engine(engine, seq_group)

        batch_sequences.add(seq)
        engine.step()
        for seq in list(batch_sequences):
            if seq.is_finished():
                batch_sequences.remove(seq)

        # If we aren't at our batch size yet, continue
        if len(batch_sequences) <= BATCH_SIZE:
            continue

        # Otherwise, loop until at least one request is done
        while not any(seq.is_finished() for seq in batch_sequences):
            engine.step()

        # Remove it from the set
        for seq in list(batch_sequences):
            if seq.is_finished():
                batch_sequences.remove(seq)

        # At this point, we are always at the case where we have finished
        # processing some number of requests from the batch after running
        # several _no_spec executions. The memory should not have
        # increased between the previous  time this was recorded and the
        # current time.
        if previous_memory_allocated is None:
            previous_memory_allocated = torch.cuda.memory_allocated()
        else:
            assert previous_memory_allocated == torch.cuda.memory_allocated()
