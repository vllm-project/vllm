import numpy as np
import pytest
import vllm
import ast

from vllm import SamplingParams

import torch
from typing import List, Optional, Tuple
from vllm.lora.request import LoRARequest
# from vllm.anyscale.tokenization import InputTooLongError

from .data.long_context_test_data import prompts_and_responses

context_len_to_scaling_factor = {
    "16k": 4,
    "32k": 8,
}

# We use the same sampling params for all requests
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=100,
)


def _create_lora_request(lora_id, long_context_infos):
    context_len = long_context_infos[lora_id]["context_length"]
    scaling_factor = context_len_to_scaling_factor[context_len]
    return LoRARequest(context_len, lora_id,
                       long_context_infos[lora_id]["lora"],
                       4096 * scaling_factor)


def evaluate_json_response(model_response, golden_response):
    """Evaluates the model response against the golden response.

    Returns a score between 0 and 1, where 1 is a perfect match and 0 is no match.
    The score quantifies how well the model is able to extract the golden JSON from the long context.
    """
    try:
        model_response = ast.literal_eval(model_response)
    except:
        raise ValueError(
            f"Model response is not a valid JSON. Expected {golden_response}, got  {model_response}"
        )

    # Normally, we would flatten the dictionary and compare the values, but in this case, we know that the dictionary is only 2 levels deep
    positive_values = 0
    total_values = 0
    # We look at all the attributes of the person that we are extracting a biography of and copmare them to the golden response
    for person_attribute, person_attribute_value in golden_response.items():
        if person_attribute in model_response:
            if type(person_attribute_value) == dict:
                for sub_attribute, sub_attribute_value in person_attribute_value.items(
                ):
                    total_values += 1
                    if sub_attribute in model_response[
                            person_attribute] and model_response[
                                person_attribute][
                                    sub_attribute] == sub_attribute_value:
                        positive_values += 1
            else:
                total_values += 1
                if model_response[person_attribute] == person_attribute_value:
                    positive_values += 1
        else:
            # We count a missing sub-dict as a single missed value.
            total_values += 1

    # Return a score between 0 and 1
    return positive_values / total_values


def generate(
    llm,
    inputs: Tuple[str, SamplingParams, Optional[LoRARequest]],
):
    prompts, sampling_param, lora_request = inputs
    outputs = llm.generate(prompts, sampling_param, lora_request=lora_request)
    return outputs[0].outputs[0].text.strip()


def batched_generate(
    llm,
    inputs: List[Tuple[str, SamplingParams, Optional[LoRARequest]]],
):
    for input in inputs:
        prompt, sampling_param, lora_req = input
        requests_data = llm._validate_and_prepare_requests(
            prompt,
            sampling_param,
            lora_request=lora_req,
        )

        # Add requests to the engine and run the engine
        for request_data in requests_data:
            llm._add_request(**request_data)
    outputs = llm._run_engine(use_tqdm=True)
    return [outputs[i].outputs[0].text.strip() for i in range(len(outputs))]


class TestLongContext:

    def _get_lora_llm(self, long_context_infos):
        scaling_factors = [
            context_len_to_scaling_factor[info["context_length"]]
            for info in long_context_infos.values()
        ]

        lora_llm = vllm.LLM(
            "meta-llama/Llama-2-13b-chat-hf",
            enable_lora=True,
            max_num_seqs=16,
            max_loras=2,
            long_lora_scaling_factors=tuple(scaling_factors),
            max_num_batched_tokens=4096 * 8,
            tensor_parallel_size=1,
            enforce_eager=True,
        )
        return lora_llm

    def test_batched_rope_kernel(self, long_context_infos):
        """We test the batched kernel by comparing the results of batched and non-batched generation."""
        lora_llm = self._get_lora_llm(long_context_infos)

        # Create non batched results first to compare against batched results
        non_batched_results = []

        for lora_id, info in long_context_infos.items():
            context_len = info["context_length"]
            lora_prompt = (prompts_and_responses[context_len][0]["prompt"],
                           sampling_params,
                           _create_lora_request(lora_id, long_context_infos))
            lora_output = generate(lora_llm, lora_prompt)
            non_batched_results.append(lora_output)

        # Create batched results
        # Each element of the batch must be (prompt, prompt_sampling_params, prompt_lora_request)
        batched_prompts = []
        for lora_id, info in long_context_infos.items():
            context_len = info["context_length"]
            batched_prompts.extend([
                (prompts_and_responses[context_len][0]["prompt"],
                 sampling_params,
                 _create_lora_request(lora_id, long_context_infos))
            ])
        batched_results = batched_generate(lora_llm, batched_prompts)

        # Results should be the same
        for non_batched, batched in zip(non_batched_results, batched_results):
            assert non_batched == batched, f"Non batched and batched results should be the same:\n{batched}\n{non_batched}"


#     def test_self_consistency(self, long_context_infos):
#         """We test consistency of the batched kernel by permuting batched inputs and comparing the results to the non-permuted batched results."""
#         lora_llm = self._get_lora_llm(long_context_infos)
#         num_loras = len(long_context_infos)

#         # Create results in order of long_context_infos
#         batched_prompts = []
#         for lora_id, info in long_context_infos.items():
#             context_len = info["context_length"]
#             batched_prompts.extend([
#                 (prompts_and_responses[context_len][0]["prompt"],
#                  sampling_params,
#                  _create_lora_request(lora_id, long_context_infos))
#             ])

#         batched_results = batched_generate(lora_llm, batched_prompts)

#         permutation = np.random.default_rng(seed=42).permutation(num_loras)

#         # Create results in random order of permutation
#         batched_prompts = []
#         for i in permutation:
#             lora_id, info = list(long_context_infos.items())[i]
#             context_len = info["context_length"]
#             batched_prompts.extend([
#                 (prompts_and_responses[context_len][0]["prompt"],
#                  sampling_params,
#                  _create_lora_request(lora_id, long_context_infos))
#             ])

#         permutated_batched_results = batched_generate(lora_llm,
#                                                       batched_prompts)

#         # Results should be the same
#         for i in range(num_loras):
#             assert batched_results[i] == permutated_batched_results[permutation[
#                 i]], f"Results should be the same:\n{batched_results[i]}\n{permutated_batched_results[permutation[i]]}"

    def test_quality(self, long_context_infos):
        """We test the quality of the answers given by the LoRA model by comparing the generated text to the merged model's outputs.

        This is effectively a mini-benchmark over four prompts.
        If this test fails, this indicates that the quality of the LoRA model is suboptimal compared to the merged model.
        For example, if the model does not output valid dictionaries, this test will fail.

        If needed for testing, the merged versions of the models are available as part of the `conftest`.
a
        The test is expected to run for about 1 minute on a p4de.24xlarge instance.
        """
        lora_llm = self._get_lora_llm(long_context_infos)

        scores = []
        for lora_id, info in long_context_infos.items():
            context_len = info["context_length"]
            for prompt_and_response in prompts_and_responses[context_len]:
                lora_prompt = (prompt_and_response["prompt"], sampling_params,
                               _create_lora_request(lora_id,
                                                    long_context_infos))
                response = generate(lora_llm, lora_prompt)
                breakpoint()
                golden_answer = prompt_and_response["golden_answer"]
                score = evaluate_json_response(response, golden_answer)
                scores.append(score)
                assert score > 0.3, f"Quality of the answer is not good enough. Expected {golden_answer}, got {response}"
        assert np.mean(scores) > 0.5

#     def test_max_len(self, long_context_infos):
#         """Test that we raise an InputTooLongError when the input of a given LoRA model exceeds the maximum length."""
#         lora_llm = self._get_lora_llm(long_context_infos)

#         # Since each LoRA model has a different maximum length, we need to test each one separately
#         for lora_id, info in long_context_infos.items():
#             context_len = info["context_length"]
#             lora_request = _create_lora_request(lora_id, long_context_infos)
#             # Good prompt should be fine
#             good_prompt = prompts_and_responses[context_len][0]["prompt"]
#             generate(lora_llm, [(good_prompt, sampling_params, lora_request)])
#             # Bad prompt should raise an error
#             bad_prompt = good_prompt * 2
#             with pytest.raises(InputTooLongError):
#                 generate(lora_llm,
#                          [(bad_prompt, sampling_params, lora_request)])

#         # Also test batched
#         batched_prompts = []
#         for lora_id_with_bad_inputs in long_context_infos.keys():
#             for lora_id, info in long_context_infos.items():
#                 context_len = info["context_length"]
#                 batched_prompts.extend([
#                     (prompts_and_responses[context_len][0]["prompt"] *
#                      (2 if lora_id == lora_id_with_bad_inputs else 1),
#                      sampling_params,
#                      _create_lora_request(lora_id, long_context_infos))
#                 ])
#             # Turn good prompt into bad prompt inside of batched prompts

#             with pytest.raises(InputTooLongError):
#                 batched_generate(lora_llm, batched_prompts)
