# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import uuid
from typing import Optional

import pytest
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.spec_decode.predicted_proposer import PredictedProposer

MODEL_NAME = "facebook/opt-125m"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "Hello my name is Bram and I love speculative decoding"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids
PREDICTED_TOKENS = TOKENIZER("test prediction").input_ids


def test_predicted_proposer():

    def predicted_proposer() -> PredictedProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model=MODEL_NAME)
        return PredictedProposer(vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                method="predicted",
                num_speculative_tokens=32,  # maximum tokens
            )))

    result = predicted_proposer().propose(context_token_ids=[[1, 2, 3, 4, 5]],
                                          predicted_tokens=[[1, 2, 3, 4, 5]])
    assert len(result[0]) == 5


class TestPredictedOutputsProposer:

    @pytest.fixture(scope="class")
    def llm_instance(self):
        """Create a single LLM instance for the entire test class"""
        llm = LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=0.1,
            speculative_config={
                "method": "predicted",
                "num_speculative_tokens": 32
            },
            max_model_len=512,
        )
        yield llm

    @pytest.fixture(scope="session")
    def engine_client_instance(self):
        engine_args = EngineArgs(model=MODEL_NAME,
                                 enforce_eager=True,
                                 gpu_memory_utilization=0.8)
        vllm_config = engine_args.create_engine_config(
            UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)

        with set_default_torch_num_threads(1):
            client = EngineCoreClient.make_client(
                multiprocess_mode=False,
                asyncio_mode=False,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )
            yield client

    def test_single_predicted_output(self, llm_instance):
        """Test single request with predicted output"""
        sampling_params = SamplingParams(predicted_outputs="test prediction",
                                         max_tokens=10,
                                         temperature=0.0)

        outputs = llm_instance.generate("Hello world", sampling_params)
        assert len(outputs) == 1
        assert outputs[0].outputs[0].text is not None

    def test_different_sampling_params_per_prompt(self, llm_instance):
        """Test different predicted_outputs for each prompt"""
        prompts = [
            "The capital of France is",
            "Python is a programming language that",
            "Machine learning is",
        ]

        sampling_params_list = [
            SamplingParams(predicted_outputs=" Paris, the beautiful city",
                           max_tokens=10,
                           temperature=0.0),
            SamplingParams(predicted_outputs=" is easy to learn and powerful",
                           max_tokens=10,
                           temperature=0.0),
            SamplingParams(
                predicted_outputs=" a field of artificial intelligence",
                max_tokens=10,
                temperature=0.0),
        ]

        outputs = llm_instance.generate(prompts, sampling_params_list)
        assert len(outputs) == len(prompts)

    def test_large_batch_mixed_params(self, llm_instance):
        """Test large batch with varied sampling parameters"""
        batch_size = 20

        prompts = [f"Request {i}:" for i in range(batch_size)]
        sampling_params_list = []

        for i in range(batch_size):
            sampling_params_list.append(
                SamplingParams(
                    predicted_outputs=f" prediction {i}",
                    max_tokens=5,
                    temperature=0.0,
                ))

        outputs = llm_instance.generate(prompts, sampling_params_list)
        assert len(outputs) == batch_size

    def test_mixed_predicted_outputs(self, llm_instance):
        """Test batch where only some prompts have predicted_outputs"""
        prompts = [
            "Prompt with prediction:",
            "Prompt without prediction:",
            "Another with prediction:",
            "Another without prediction:",
        ]

        sampling_params_list = [
            SamplingParams(predicted_outputs=" this has a prediction",
                           max_tokens=5),
            SamplingParams(predicted_outputs=None, max_tokens=5),
            SamplingParams(predicted_outputs=" another prediction",
                           max_tokens=5),
            SamplingParams(predicted_outputs=None, max_tokens=5),
        ]

        outputs = llm_instance.generate(prompts, sampling_params_list)
        assert len(outputs) == 4

    def make_request(self,
                     params: SamplingParams,
                     prompt_tokens_ids: Optional[list[int]] = None,
                     req_id: str = '') -> EngineCoreRequest:
        if not prompt_tokens_ids:
            prompt_tokens_ids = PROMPT_TOKENS

        return EngineCoreRequest(
            request_id=req_id if req_id else str(uuid.uuid4()),
            prompt_token_ids=prompt_tokens_ids,
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

    def test_client_add_request_codepath_directly(self,
                                                  engine_client_instance):
        """Directly test the add_request code path"""
        sampling_params = SamplingParams(predicted_outputs=PREDICTED_TOKENS,
                                         max_tokens=5)

        requests = [self.make_request(sampling_params) for _ in range(10)]
        request_ids = [req.request_id for req in requests]
        for request in requests:
            engine_client_instance.add_request(request)
            time.sleep(0.01)

        # Step until completion
        outputs: dict[str, list] = {req_id: [] for req_id in request_ids}
        while True:
            engine_core_outputs = engine_client_instance.get_output().outputs
            if len(engine_core_outputs) == 0:
                continue
            all_finished = True
            for out in engine_core_outputs:
                outputs[out.request_id].append(out)
                if not out.finished:
                    all_finished = False
            if all_finished:
                break

    def test_client_add_request_multiple_individual(self,
                                                    engine_client_instance):
        """Test multiple individual add_request calls"""
        request_ids = []
        for i in range(5):
            request_id = f"request_{i}"
            predicted_outputs = TOKENIZER(f"prediction {i}").input_ids
            sampling_params = SamplingParams(
                predicted_outputs=predicted_outputs, max_tokens=5)

            engine_client_instance.add_request(
                self.make_request(sampling_params, req_id=request_id))
            request_ids.append(request_id)

        # Step until completion
        outputs: dict[str, list] = {req_id: [] for req_id in request_ids}
        while True:
            engine_core_outputs = engine_client_instance.get_output().outputs
            if len(engine_core_outputs) == 0:
                continue
            all_finished = True
            for out in engine_core_outputs:
                outputs[out.request_id].append(out)
                if not out.finished:
                    all_finished = False
            if all_finished:
                break

    @pytest.mark.parametrize("batch_size", [1, 3, 5])
    def test_various_batch_sizes(self, llm_instance, batch_size):
        """Test with different batch sizes"""
        prompts = ["Hello world"] * batch_size
        sampling_params = SamplingParams(
            predicted_outputs="this is a prediction", max_tokens=5)

        outputs = llm_instance.generate(prompts, sampling_params)
        assert len(outputs) == batch_size

    def test_empty_predicted_outputs(self, llm_instance):
        """Test edge cases with predicted outputs"""
        test_cases = [
            ("", "Empty string"),
            (None, "None value"),
            ("   ", "Whitespace only"),
        ]

        for predicted_output, description in test_cases:
            sampling_params = SamplingParams(
                predicted_outputs=predicted_output, max_tokens=5)

            outputs = llm_instance.generate("Test prompt", sampling_params)
            assert len(outputs) == 1

    def test_very_long_predicted_outputs(self, llm_instance):
        """Test with very long predicted outputs"""
        long_prediction = "This is a very long prediction " * 20  # 600+ chars

        sampling_params = SamplingParams(predicted_outputs=long_prediction,
                                         max_tokens=5)

        outputs = llm_instance.generate("Test:", sampling_params)
        assert len(outputs) == 1

    def test_special_characters_predicted_outputs(self, llm_instance):
        """Test predicted outputs with special characters"""
        special_predictions = [
            "Hello! 🌟 How are you?",  # Emojis
            "Code: print('hello')",  # Code
            "Math: 2 + 2 = 4",  # Numbers
            "Symbols: @#$%^&*()",  # Special chars
        ]

        for prediction in special_predictions:
            sampling_params = SamplingParams(predicted_outputs=prediction,
                                             max_tokens=5)

            outputs = llm_instance.generate("Input:", sampling_params)
            assert len(outputs) == 1
