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

RTOL = 0.06
TEST_DATA_FILE = os.environ.get(
    "LM_EVAL_TEST_DATA_FILE",
    ".jenkins/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct.yaml")

REPORT_PERFORMANCE = os.environ.get("LM_EVAL_REPORT_PERFORMANCE",
                                    "false") in ['1', 'true']

TP_SIZE = os.environ.get("LM_EVAL_TP_SIZE", 1)


def setup_fp8():
    os.environ[
        "QUANT_CONFIG"] = \
            "inc_unit_scales_config.json"


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
    if eval_config.get("fp8"):
        model_args += ",quantization=inc," \
            "kv_cache_dtype=fp8_inc," \
            "weights_load_device=cpu"
    if eval_config.get("num_scheduler_steps"):
        model_args += \
            f",num_scheduler_steps={eval_config.get('num_scheduler_steps')}"
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


def report_performance(task, input_lens, output_lens, time, record_property):
    assert len(input_lens) == len(output_lens)
    context_lens = [i + o for i, o in zip(input_lens, output_lens)]
    gen_tput = sum(output_lens) / time
    all_lens = [input_lens, output_lens, context_lens]
    min_input_tokens, min_output_tokens, min_context_tokens = (
        min(x) for x in all_lens)
    max_input_tokens, max_output_tokens, max_context_tokens = (
        max(x) for x in all_lens)
    mean_input_tokens, mean_output_tokens, mean_context_tokens = (
        statistics.mean(x) for x in all_lens)
    stddev_input_tokens, stddev_output_tokens, stddev_context_tokens = (
        statistics.stdev(x) for x in all_lens)
    msg = (
        f'{task} | estimated average generation throughput: {gen_tput:.2f} tokens/s \n'  # noqa: G004, E501
        f'{task} | input_tokens   | min: {min_input_tokens} | max: {max_input_tokens} | mean: {mean_input_tokens:.2f} | stddev: {stddev_input_tokens:.2f}\n'  # noqa: E501
        f'{task} | output_tokens  | min: {min_output_tokens} | max: {max_output_tokens} | mean: {mean_output_tokens:.2f} | stddev: {stddev_output_tokens:.2f}\n'  # noqa: E501
        f'{task} | context_length | min: {min_context_tokens} | max: {max_context_tokens} | mean: {mean_context_tokens:.2f} | stddev: {stddev_context_tokens:.2f}'  # noqa: E501
    )

    # Log all of these stats to JUnitXML
    record_property(f"{task}_gen_tput", gen_tput)
    record_property(f"{task}_input_tokens_min", min_input_tokens)
    record_property(f"{task}_input_tokens_max", max_input_tokens)
    record_property(f"{task}_input_tokens_mean", mean_input_tokens)
    record_property(f"{task}_input_tokens_stddev", stddev_input_tokens)

    record_property(f"{task}_output_tokens_min", min_output_tokens)
    record_property(f"{task}_output_tokens_max", max_output_tokens)
    record_property(f"{task}_output_tokens_mean", mean_output_tokens)
    record_property(f"{task}_output_tokens_stddev", stddev_output_tokens)

    record_property(f"{task}_context_tokens_min", min_context_tokens)
    record_property(f"{task}_context_tokens_max", max_context_tokens)
    record_property(f"{task}_context_tokens_mean", mean_context_tokens)
    record_property(f"{task}_context_tokens_stddev", stddev_context_tokens)

    print(msg)


def get_current_gaudi_platform():
    """
    Inspired by: https://github.com/HabanaAI/Model-References/blob/a87c21f14f13b70ffc77617b9e80d1ec989a3442/PyTorch/computer_vision/classification/torchvision/utils.py#L274
    """
    import habana_frameworks.torch.utils.experimental as htexp

    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "Gaudi1"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "Gaudi2"
    elif device_type == htexp.synDeviceType.synDeviceGaudi3:
        return "Gaudi3"
    else:
        raise ValueError(
            f"Unsupported device: the device type is {device_type}.")


def test_lm_eval_correctness(record_xml_attribute, record_property):
    try:
        eval_config = yaml.safe_load(
            Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

        # Record JUnitXML test name
        tasks_str = '_'.join([t['name'] for t in eval_config["tasks"]])
        platform = get_current_gaudi_platform()
        testname = (f'test_{Path(TEST_DATA_FILE).stem}_{tasks_str}_{platform}_'
                    f'tp{TP_SIZE}')
        record_xml_attribute("name", testname)

        # Set up environment for FP8 inference
        if eval_config.get("fp8"):
            setup_fp8()
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
                        tokenizer(
                            list(itertools.chain.from_iterable(
                                x['resps'])))['input_ids'])) for x in samples
            ]
            tokenized_outputs_lens = [len(x) for x in tokenized_outputs]
            if REPORT_PERFORMANCE:
                report_performance(task['name'], tokenized_inputs_lens,
                                   tokenized_outputs_lens, total_time,
                                   record_property)

            for metric in task["metrics"]:
                ground_truth = metric["value"]
                measured_value = results["results"][task["name"]][
                    metric["name"]]
                print(
                    f'{task["name"]} | {metric["name"]}: '
                    f'ground_truth={ground_truth} | measured={measured_value}')

                # Record ground truth and measured value to JUnitXML
                record_property(
                    f"{task['name']}_{metric['name']}_ground_truth",
                    ground_truth)
                record_property(f"{task['name']}_{metric['name']}_measured",
                                measured_value)
                assert numpy.isclose(ground_truth, measured_value, rtol=RTOL)
    except Exception as exc:
        # nasty workaround for a nasty HPU PT bridge bug (SW-204785)
        atexit.register(fail_on_exit)
        raise exc
