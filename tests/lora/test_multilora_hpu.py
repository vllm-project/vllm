from multiprocessing import Process
from typing import List, Optional, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    """
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None),
        ("To be or not to be,",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        max_tokens=128), None),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora2", 2, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    result = {}

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                result[
                    request_output.request_id] = request_output.outputs[0].text
    return result


expected_output = [
    " or, through inaction, allow a human being to come to harm.\nA robot must obey the orders given it by human beings except where such orders would conflict with the First Law.\nA robot must protect its own existence as long as such protection does not conflict with the First or Second Law.\nThe Three Laws of Robotics were created by Isaac Asimov in 1942. They are the foundation of robotics and artificial intelligence.\nThe Three Laws of Robotics are the foundation of robotics and artificial intelligence. They were created by Isaac Asimov in 194",  # noqa: E501
    " that is the question.\nIt is the most famous line in all of Shakespeare's plays and one of the most famous in English literature. The question is not whether or not to be, but rather the question of who to be.\nIn Hamlet's case, the question is whether or not to be a good person. He is torn between the goodness of his father and the evil of his mother.\nThe question is a difficult one, and one that has been asked many times before. In Hamlet's case, the question is whether or not to be a good person, and he is torn between the",  # noqa: E501
    "  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",  # noqa: E501
    "  SELECT nationality FROM table_name_11 WHERE elector = 'anchero pantaleone' ",  # noqa: E501
    "  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",  # noqa: E501
    "  SELECT nationality FROM table_name_11 WHERE elector = 'anchero pantaleone' "  # noqa: E501
]


def _test_llama_multilora(sql_lora_files, tp_size):
    """Main function that sets up and runs the prompt processing."""
    engine_args = EngineArgs(
        model="/mnt/weka/data/pytorch/llama2/Llama-2-7b-hf",
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_num_seqs=256,
        dtype='float32',
        tensor_parallel_size=tp_size)
    engine = LLMEngine.from_engine_args(engine_args)
    test_prompts = create_test_prompts(sql_lora_files)
    results = process_requests(engine, test_prompts)
    generated_texts = [results[key] for key in sorted(results)]
    assert generated_texts == expected_output


def test_llama_multilora_1x(sql_lora_files):
    # Work-around to resolve stalling issue in multi-card scenario
    p = Process(target=_test_llama_multilora, args=(sql_lora_files, 1))
    p.start()
    p.join()
    assert p.exitcode == 0


def test_llama_multilora_2x(sql_lora_files):
    # Work-around to resolve stalling issue in multi-card scenario
    p = Process(target=_test_llama_multilora, args=(sql_lora_files, 2))
    p.start()
    p.join()
    assert p.exitcode == 0


def test_llama_multilora_4x(sql_lora_files):
    # Work-around to resolve stalling issue in multi-card scenario
    p = Process(target=_test_llama_multilora, args=(sql_lora_files, 4))
    p.start()
    p.join()
    assert p.exitcode == 0
