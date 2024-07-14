import pytest

from tests.models.utils import check_outputs_equal
from vllm.worker.model_runner import _get_graph_batch_size

MODELS = ["ai21labs/Jamba-tiny-random"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [20])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # To pass the small model tests, we need full precision.
    assert dtype == "float"

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
        hf_logprobs_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs=2)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        vllm_logprobs_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs=2)

    for i in range(len(example_prompts)):
        _, hf_output_str = hf_outputs[i]
        hf_output_ids, _, hf_output_logprobs = hf_logprobs_outputs[i]

        _, vllm_output_str = vllm_outputs[i]
        vllm_output_ids, _, vllm_output_logprobs = vllm_logprobs_outputs[i]

        if hf_output_str != vllm_output_str:
            first_diff_index = [
                hf_id == vllm_id
                for hf_id, vllm_id in zip(hf_output_ids, vllm_output_ids)
            ].index(False)
            hf_disagreement_logprobs = hf_output_logprobs[first_diff_index]
            vllm_disagreement_logprobs = {
                k: v.logprob
                for k, v in vllm_output_logprobs[first_diff_index].items()
            }

            assert (hf_output_ids[first_diff_index]
                    in vllm_disagreement_logprobs), (
                        f"Test{i}:different outputs\n"
                        f"HF: {hf_output_str!r}\n"
                        f"vLLM: {vllm_output_str!r}\n",
                        f"Disagreement in {first_diff_index}th token. "
                        f"HF id: {hf_output_ids[first_diff_index]}, "
                        f"vLLM id: {vllm_output_ids[first_diff_index]})\n",
                        "HF top token not in vLLM top 2 tokens")

            vllm_disagreement_logprobs_values = list(
                vllm_disagreement_logprobs.values())
            vllm_logprobs_diff = abs(vllm_disagreement_logprobs_values[0] -
                                     vllm_disagreement_logprobs_values[1])
            vllm_hf_diff = abs(
                hf_disagreement_logprobs[hf_output_ids[first_diff_index]] -
                vllm_disagreement_logprobs[hf_output_ids[first_diff_index]])

            assert (vllm_logprobs_diff < vllm_hf_diff
                    or vllm_logprobs_diff < 1e-4), (
                        f"Test{i}:different outputs\n"
                        f"HF: {hf_output_str!r}\n"
                        f"vLLM: {vllm_output_str!r}\n",
                        f"Disagreement in {first_diff_index}th token. "
                        f"HF id: {hf_output_ids[first_diff_index]}, "
                        f"vLLM id: {vllm_output_ids[first_diff_index]})\n",
                        f"HF top token in vLLM top 2 tokens, "
                        f"but logprobs diff is too large. "
                        f"vLLM top 2 logprob diff: {vllm_logprobs_diff}\n",
                        f"HF to vLLM diff of top HF token: {vllm_hf_diff}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [15])
def test_batching(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # To pass the small model tests, we need full precision.
    # assert dtype == "float"

    with vllm_runner(model, dtype=dtype, enforce_eager=True) as vllm_model:
        for_loop_outputs = []
        for_loop_logprobs_outputs = []
        for prompt in example_prompts:
            for_loop_outputs.append(
                vllm_model.generate_greedy([prompt], max_tokens)[0])
            for_loop_logprobs_outputs.append(
                vllm_model.generate_greedy_logprobs([prompt],
                                                    max_tokens,
                                                    num_logprobs=2)[0])

        batched_outputs = vllm_model.generate_greedy(example_prompts,
                                                     max_tokens)
        batched_logprobs_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs=2)

    for i in range(len(example_prompts)):
        _, for_loop_output_str = for_loop_outputs[i]
        (for_loop_output_ids, _,
         for_loop_output_logprobs) = for_loop_logprobs_outputs[i]

        _, batched_output_str = batched_outputs[i]
        (batched_output_ids, _,
         batched_output_logprobs) = batched_logprobs_outputs[i]

        if for_loop_output_str != batched_output_str:
            first_diff_index = [
                for_loop_id == batched_id for for_loop_id, batched_id in zip(
                    for_loop_output_ids, batched_output_ids)
            ].index(False)
            for_loop_disagreement_logprobs = {
                k: v.logprob
                for k, v in for_loop_output_logprobs[first_diff_index].items()
            }
            batched_disagreement_logprobs = {
                k: v.logprob
                for k, v in batched_output_logprobs[first_diff_index].items()
            }

            assert (
                for_loop_output_ids[first_diff_index]
                in batched_disagreement_logprobs), (
                    f"Test{i}:different outputs\n"
                    f"For-loop: {for_loop_output_str!r}\n",
                    f"Batched: {batched_output_str!r}\n",
                    f"Disagreement in {first_diff_index}th token. "
                    f"For-loop id: {for_loop_output_ids[first_diff_index]}, "
                    f"Batched id: {batched_output_ids[first_diff_index]})\n",
                    "For-loop top token not in batched top 2 tokens")

            batched_disagreement_logprobs_values = list(
                batched_disagreement_logprobs.values())
            batched_logprobs_diff = abs(
                batched_disagreement_logprobs_values[0] -
                batched_disagreement_logprobs_values[1])
            batched_for_loop_diff = abs(
                for_loop_disagreement_logprobs[
                    for_loop_output_ids[first_diff_index]] -
                batched_disagreement_logprobs[
                    for_loop_output_ids[first_diff_index]])

            assert (
                batched_logprobs_diff < batched_for_loop_diff
                or batched_logprobs_diff < 1e-4), (
                    f"Test{i}:different outputs\n"
                    f"For-loop: {for_loop_output_str!r}\n"
                    f"Batched: {batched_output_str!r}\n",
                    f"Disagreement in {first_diff_index}th token. "
                    f"For-loop id: {for_loop_output_ids[first_diff_index]}, "
                    f"Batched id: {batched_output_ids[first_diff_index]})\n",
                    f"For-loop top token in batched top 2 tokens, "
                    f"but logprobs diff is too large. "
                    f"Batched top 2 logprob diff: {batched_logprobs_diff}\n",
                    f"For-loop to batched diff of top for-loop token: "
                    f"{batched_logprobs_diff}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [20])
def test_mamba_cache_cg_padding(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # This test is for verifying that mamba cache is padded to CG captured
    # batch size. If it's not, a torch RuntimeError will be raised because
    # tensor dimensions aren't compatible
    while len(example_prompts) == _get_graph_batch_size(len(example_prompts)):
        example_prompts.append(example_prompts[0])

    try:
        with vllm_runner(model, dtype=dtype) as vllm_model:
            vllm_model.generate_greedy(example_prompts, max_tokens)
    except RuntimeError:
        pytest.fail(
            "Couldn't run batch size which is not equal to a Cuda Graph "
            "captured batch size. "
            "Could be related to mamba cache not padded correctly")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_models_preemption_recompute(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # Tests that outputs are identical with and w/o preemtions (recompute)
    assert dtype == "float"

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_model.model.llm_engine.scheduler[
            0].ENABLE_ARTIFICIAL_PREEMPT = True
        preempt_vllm_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

        vllm_model.model.llm_engine.scheduler[
            0].ENABLE_ARTIFICIAL_PREEMPT = False
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=preempt_vllm_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="vllm_preepmtions",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_fail_upon_inc_requests_and_finished_requests_lt_available_blocks(
    vllm_runner,
    model: str,
    dtype: str,
    example_prompts,
) -> None:
    # This test is for verifying that the Jamba inner state management doesn't
    # collapse in case where the number of incoming requests and
    # finished_requests_ids is larger than the maximum mamba block capacity.
    # This could generally happen due to the fact that Jamba does support
    # statelessness mechanism where it can cleanup new incoming requests in
    # a single step.
    try:
        with vllm_runner(model, dtype=dtype, max_num_seqs=10) as vllm_model:
            vllm_model.generate_greedy([example_prompts[0]] * 100, 10)
    except ValueError:
        pytest.fail("Jamba inner state wasn't cleaned up properly between"
                    "steps finished requests registered unnecessarily ")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_state_cleanup(
    vllm_runner,
    model: str,
    dtype: str,
    example_prompts,
) -> None:
    # This test is for verifying that the Jamba state is cleaned up between
    # steps, If its not cleaned, an error would be expected.
    try:
        with vllm_runner(model, dtype=dtype) as vllm_model:
            for _ in range(10):
                vllm_model.generate_greedy([example_prompts[0]] * 100, 1)
    except ValueError:
        pytest.fail("Jamba inner state wasn't cleaned up between states, "
                    "could be related to finished_requests_ids")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_model_print(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(vllm_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
