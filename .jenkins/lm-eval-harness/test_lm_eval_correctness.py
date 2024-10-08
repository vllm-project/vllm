"""
LM eval harness on model to compare vs HF baseline computed offline.
Configs are found in configs/$MODEL.yaml

* export LM_EVAL_TEST_DATA_FILE=configs/Meta-Llama-3-70B-Instruct.yaml
* export LM_EVAL_TP_SIZE=4 
* pytest -s test_lm_eval_correctness.py
"""
import atexit
import itertools
import os
import statistics
import time
from pathlib import Path

import lm_eval
import numpy
import yaml

import vllm

RTOL = 0.05
TEST_DATA_FILE = os.environ.get(
    "LM_EVAL_TEST_DATA_FILE",
    ".jenkins/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct.yaml")

TP_SIZE = os.environ.get("LM_EVAL_TP_SIZE", 1)


def fail_on_exit():
    os._exit(1)


def launch_lm_eval(eval_config):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    dtype = eval_config.get('dtype', 'bfloat16')
    max_num_seqs = eval_config.get('max_num_seqs', 128)
    model_args = f"pretrained={eval_config['model_name']}," \
                 f"tensor_parallel_size={TP_SIZE}," \
                 f"add_bos_token=true," \
                 f"dtype={dtype}," \
                 f"max_model_len=4096," \
                 f"max_num_seqs={max_num_seqs}," \
                 f"trust_remote_code={trust_remote_code}"
    kwargs = {}
    if 'fewshot_as_multiturn' in eval_config:
        kwargs['fewshot_as_multiturn'] = eval_config['fewshot_as_multiturn']
    if 'apply_chat_template' in eval_config:
        kwargs['apply_chat_template'] = eval_config['apply_chat_template']
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        batch_size="auto",
        **kwargs)

    return results


def report_performance(task, input_lens, output_lens, time):
    assert len(input_lens) == len(output_lens)
    context_lens = [i + o for i, o in zip(input_lens, output_lens)]
    gen_tput = sum(output_lens) / time
    msg = (
        f'{task} | average generation throughput: {gen_tput:.2f} tokens/s \n'  # noqa: G004
        f'{task} | input_tokens   | min: {min(input_lens)} | max: {max(input_lens)} | mean: {statistics.mean(input_lens):.2f} | stddev: {statistics.stdev(input_lens):.2f}\n'  # noqa: E501
        f'{task} | output_tokens  | min: {min(output_lens)} | max: {max(output_lens)} | mean: {statistics.mean(output_lens):.2f} | stddev: {statistics.stdev(output_lens):.2f}\n'  # noqa: E501
        f'{task} | context_length | min: {min(context_lens)} | max: {max(context_lens)} | mean: {statistics.mean(context_lens):.2f} | stddev: {statistics.stdev(context_lens):.2f}'  # noqa: E501
    )
    print(msg)


def test_lm_eval_correctness():
    eval_config = yaml.safe_load(
        Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

    # Launch eval requests.
    start_time = time.perf_counter()
    results = launch_lm_eval(eval_config)
    total_time = time.perf_counter() - start_time

    tokenizer = vllm.transformers_utils.tokenizer.get_tokenizer(
        eval_config['model_name'])

    # Confirm scores match ground truth.
    for task in eval_config["tasks"]:

        samples = results['samples'][task["name"]]
        tokenized_inputs = [
            tokenizer(x['arguments'][0][0])['input_ids'] for x in samples
        ]
        tokenized_inputs_lens = [len(x) for x in tokenized_inputs]
        tokenized_outputs = [
            list(
                itertools.chain.from_iterable(
                    tokenizer(list(itertools.chain.from_iterable(
                        x['resps'])))['input_ids'])) for x in samples
        ]
        tokenized_outputs_lens = [len(x) for x in tokenized_outputs]
        report_performance(task['name'], tokenized_inputs_lens,
                           tokenized_outputs_lens, total_time)

        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(f'{task["name"]} | {metric["name"]}: '
                  f'ground_truth={ground_truth} | measured={measured_value}')
            try:
                assert numpy.isclose(ground_truth, measured_value, rtol=RTOL)
            except AssertionError as exc:
                # nasty workaround for HPU PT bridge bug (SW-204785)
                atexit.register(fail_on_exit)
                raise exc
