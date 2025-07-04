# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.optimum_neuron import (
    OptimumNeuronModelForCausalLM)
from vllm.platforms import current_platform
from vllm.platforms.neuron import NeuronFramework
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.worker.optimum_neuron_model_runner import (
    ModelInputForOptimumNeuron, OptimumNeuronModelRunner)

os.environ['VLLM_NEURON_FRAMEWORK'] = NeuronFramework.OPTIMUM_NEURON.value


def _create_neuron_model_runner(model: str, *args,
                                **kwargs) -> OptimumNeuronModelRunner:
    engine_args = EngineArgs(model, *args, **kwargs)
    engine_config = engine_args.create_engine_config()
    vllm_config = VllmConfig(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
    )
    neuron_model_runner = OptimumNeuronModelRunner(vllm_config=vllm_config)
    return neuron_model_runner


def test_optimum_neuron_runner_create():
    if not current_platform.use_optimum_neuron():
        pytest.skip("This test is only relevant when using"
                    " the optimum-neuron framework.")

    # Instantiate an empty model runner
    model_runner = _create_neuron_model_runner(
        "unsloth/Llama-3.2-1B-Instruct",
        seed=0,
        max_num_seqs=4,
        max_model_len=4096,
        tensor_parallel_size=2,
        dtype="bfloat16",
    )

    # Load model
    model_runner.load_model()
    model = model_runner.get_model()
    assert isinstance(model, OptimumNeuronModelForCausalLM)

    # Prepare inputs
    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id="test_0",
            is_prompt=True,
            seq_data={0: SequenceData.from_seqs([1, 2, 3])},
            sampling_params=SamplingParams(temperature=0.5, top_k=1,
                                           top_p=0.5),
            block_tables={0: [1]},
        ),
        SequenceGroupMetadata(
            request_id="test_0",
            is_prompt=True,
            seq_data={1: SequenceData.from_seqs([4, 5, 6])},
            sampling_params=SamplingParams(temperature=0.2, top_k=2,
                                           top_p=0.2),
            block_tables={1: [0]},
        )
    ]

    inputs = model_runner.prepare_model_input(seq_group_metadata_list)
    assert isinstance(inputs, ModelInputForOptimumNeuron)

    # Execute the model
    outputs = model_runner.execute_model(inputs)
    assert len(outputs) == 1
    assert isinstance(outputs[0], SamplerOutput)
