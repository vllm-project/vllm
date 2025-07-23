# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from unittest.mock import MagicMock

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.platforms.neuron import NeuronFramework
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.worker.neuron_model_runner import NeuronModelRunner

os.environ[
    'VLLM_NEURON_FRAMEWORK'] = NeuronFramework.TRANSFORMERS_NEURONX.value


def _create_neuron_model_runner(model: str, *args,
                                **kwargs) -> NeuronModelRunner:
    engine_args = EngineArgs(model, *args, **kwargs)
    engine_config = engine_args.create_engine_config()
    vllm_config = VllmConfig(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
    )
    neuron_model_runner = NeuronModelRunner(vllm_config=vllm_config)
    return neuron_model_runner


def test_update_neuron_sampling_params_not_full_batch():
    os.environ["NEURON_ON_DEVICE_SAMPLING_DISABLED"] = "0"
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=2,
    )
    assert not model_runner._on_device_sampling_disabled
    # Test sampling param updating only when TNx is framework
    # NxDI handles sampling parameter updating inside model
    if current_platform.use_transformers_neuronx():
        model_mock = MagicMock()
        model_runner.model = model_mock

        seq_group_metadata_list = [
            SequenceGroupMetadata(
                request_id="test_0",
                is_prompt=True,
                seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0.5,
                                               top_k=1,
                                               top_p=0.5),
                block_tables={0: [1]},
            )
        ]

        model_runner.prepare_model_input(seq_group_metadata_list)

        # Index neuron sampling parameters based on block_tables indices.
        # The first block_id of the sequence 0 is 1, so its parameters are
        # placed at index 1. So the sampling parameters will be:
        # Index 0: default sampling parameters
        # Index 1: sequecne 0's sampling parameters.
        neuron_sampling_params = (
            model_runner.model_config.neuron_sampling_params)
        assert neuron_sampling_params.temperature == [1.0, 0.5]
        assert neuron_sampling_params.top_k == [
            model_runner._MAX_NEURON_SAMPLING_TOP_K, 1
        ]
        assert neuron_sampling_params.top_p == [1.0, 0.5]
        model_mock.model.update_generation_config.assert_called_once_with(
            neuron_sampling_params)


def test_update_neuron_sampling_params_full_batch():
    os.environ["NEURON_ON_DEVICE_SAMPLING_DISABLED"] = "0"
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=2,
    )
    assert not model_runner._on_device_sampling_disabled

    # Test sampling param updating only when TNx is framework
    # NxDI handles sampling parameter updating inside model
    if current_platform.use_transformers_neuronx():
        model_mock = MagicMock()
        model_runner.model = model_mock

        seq_group_metadata_list = [
            SequenceGroupMetadata(
                request_id="test_0",
                is_prompt=True,
                seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0.5,
                                               top_k=1,
                                               top_p=0.5),
                block_tables={0: [1]},
            ),
            SequenceGroupMetadata(
                request_id="test_0",
                is_prompt=True,
                seq_data={1: SequenceData.from_seqs([4, 5, 6])},
                sampling_params=SamplingParams(temperature=0.2,
                                               top_k=2,
                                               top_p=0.2),
                block_tables={1: [0]},
            )
        ]

        model_runner.prepare_model_input(seq_group_metadata_list)

        # Index neuron sampling parameters based on block_tables indices.
        # The first block_id of the sequence 0 is 1, so its parameters are
        # placed at index 1. So the sampling parameters will be:
        # Index 0: sequence 1's sampling parameters
        # Index 1: sequecne 0's sampling parameters.
        neuron_sampling_params = (
            model_runner.model_config.neuron_sampling_params)
        assert neuron_sampling_params.temperature == [0.2, 0.5]
        assert neuron_sampling_params.top_k == [2, 1]
        assert neuron_sampling_params.top_p == [0.2, 0.5]
        model_mock.model.update_generation_config.assert_called_once_with(
            neuron_sampling_params)
