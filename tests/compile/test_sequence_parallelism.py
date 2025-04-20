# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

ALL_REDUCE_OP = "torch.ops.vllm.all_reduce.default"
ALL_GATHER_OP = "torch.ops.vllm.all_gather.default"
REDUCE_SCATTER_OP = "torch.ops.vllm.reduce_scatter.default"


def count_comm_ops(graph_path):
    all_reduce_cnt = 0
    all_gather_cnt = 0
    reduce_scatter_cnt = 0
    try:
        with open(graph_path) as f:
            for line in f:
                if ALL_REDUCE_OP in line:
                    all_reduce_cnt += 1
                if ALL_GATHER_OP in line:
                    all_gather_cnt += 1
                if REDUCE_SCATTER_OP in line:
                    reduce_scatter_cnt += 1
    except FileNotFoundError:
        print(f"Error: File '{graph_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")
    return all_reduce_cnt, all_gather_cnt, reduce_scatter_cnt


def test_sequence_parallelism_compilation():
    temp_dir = tempfile.mkdtemp()

    config = CompilationConfig(
        level=3,
        custom_ops=["+rms_norm"],
        compile_sizes=[4, 8],
        splitting_ops=[],
    )
    config.pass_config.enable_sequence_parallelism = True
    config.pass_config.dump_graph_dir = Path(temp_dir)
    config.pass_config.dump_graph_stages = \
        ["before_sequence_parallelism_pass", "after_sequence_parallelism_pass"]

    sampling_params = SamplingParams(temperature=0, )

    llm = LLM(model="unsloth/Llama-3.2-1B-Instruct",
              enforce_eager=False,
              tensor_parallel_size=2,
              dtype=torch.float16,
              max_num_batched_tokens=2048,
              compilation_config=config)

    prompts = [
        "Can you calculate 19 + 20?", "How to make a cake?",
        "How old a baby can start to try solid food?",
        "What's pros and cons of using a pacifier for baby?"
    ]

    answers = [
        " I'll let you know if you're correct", " A step-by-step guide",
        " Most pediatricians recommend ", " The American Academy of Pediatrics"
    ]

    outputs = llm.generate(prompts, sampling_params)
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text.startswith(answer)

    before_graph = os.path.join(temp_dir,
                                "before_sequence_parallelism_pass-0.py")
    c1, c2, c3 = count_comm_ops(before_graph)
    assert c1 > 0, "Expected all_reduce ops, but found 0 before \
        apply sequence parallelism pass"
    assert c2 == 0, f"Expected 0 all_gather ops, but found {c2} before" + \
        "apply sequence parallelism pass"
    assert c3 == 0, f"Expected 0 reduce_scatter ops, but found {c3} before" + \
        "apply sequence parallelism pass"

    after_graph = os.path.join(temp_dir,
                               "after_sequence_parallelism_pass-0.py")
    c1, c2, c3 = count_comm_ops(after_graph)

    assert c1 == 0, f"Expected 0 all_reduce ops, but found {c1} after" + \
        "apply sequence parallelism pass"
    assert c2 > 0, "Expected all_gather ops, but found 0 in after" + \
        "apply sequence parallelism pass"
    assert c3 > 0, "Expected 0 reduce_scatter ops, but found 0 after \
        apply sequence parallelism pass"

    assert c2 == c3, f"Expected all_gather ops and reduce_scatter ops to be \
        equal, but found {c2} and {c3} after apply sequence parallelism pass"
