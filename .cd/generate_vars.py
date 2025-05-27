#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pandas as pd


def get_device_model():
    import habana_frameworks.torch.hpu as hthpu
    os.environ["LOG_LEVEL_ALL"] = "6"
    HPU_determined = hthpu.get_device_name()
    return HPU_determined


def vllm_auto_calc(fd):
    tensor_parallel_size_new = max(1, min(8, fd['tensor_parallel_size']))
    if tensor_parallel_size_new != fd['tensor_parallel_size']:
        print(f"Clamping tensor_parallel_size to {tensor_parallel_size_new}")
    fd['tensor_parallel_size'] = tensor_parallel_size_new

    fd['max_model_len'] = max(1, fd['max_model_len'])

    if fd['tensor_parallel_size'] > 1:
        fd['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = True
    else:
        fd['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = False

    fd['model_mem_from_config'] = float(fd.get('model_mem_from_config'))
    fd['dtype'] = dtype
    fd['device_hpu_mem'] = hpu_mem[HPU_determined]

    print(f"{HPU_determined} Device detected with "
          f"{fd['device_hpu_mem']} GB memory.")

    fd['total_gpu_mem'] = fd['device_hpu_mem'] * fd['tensor_parallel_size']
    fd['model_mem_in_gb'] = (fd['model_mem_from_config'] * fd['quant_dtype'] /
                             fd['model_dtype']) / (1024 * 1024 * 1024)
    fd['usable_mem'] = ((fd['total_gpu_mem'] / fd['tensor_parallel_size']) -
                        fd['unavailable_mem_abs'] -
                        (fd['model_mem_in_gb'] / fd['tensor_parallel_size']) -
                        fd['profiler_mem_overhead'])
    if fd['usable_mem'] < 0:
        raise ValueError(
            f"Not enough memory for model '{os.environ['MODEL']}', "
            "increase tensor_parallel_size.")
    else:
        print(f"Usable graph+kvcache memory {fd['usable_mem']:.2f} GB")

    fd['gpu_memory_util_temp'] = (1 -
                                  fd['gpu_free_mem_target'] / fd['usable_mem'])
    fd['gpu_memory_utilization'] = math.floor(
        fd['gpu_memory_util_temp'] * 100) / 100
    fd['kv_cache_per_seq'] = (
        (2 * fd['max_model_len'] * fd['num_hidden_layers'] * fd['hidden_size']
         * fd['num_key_value_heads'] * fd['cache_dtype_bytes']) /
        fd['num_attention_heads']) / (1024 * 1024 * 1024)
    fd['est_max_num_seqs'] = (fd['tensor_parallel_size'] * fd['usable_mem'] *
                              fd['gpu_memory_utilization'] /
                              fd['kv_cache_per_seq'])
    if fd['est_max_num_seqs'] < 1:
        raise ValueError(
            "Not enough memory for kv cache. "
            "Increase tensor_parallel_size or reduce max_model_len")
    print(f"Estimating graph memory for "
          f"{fd['est_max_num_seqs']:.0f} max_num_seqs")

    fd['est_hpu_blocks'] = (fd['max_model_len'] * fd['est_max_num_seqs'] /
                            fd['block_size'])
    fd['decode_bs_ramp_graphs'] = 1 + int(
        math.log(
            fd['VLLM_DECODE_BS_BUCKET_STEP'] / fd['VLLM_DECODE_BS_BUCKET_MIN'],
            2,
        ))
    fd['decode_bs_step_graphs'] = max(
        0,
        int(1 + (fd['est_max_num_seqs'] - fd['VLLM_DECODE_BS_BUCKET_STEP']) /
            fd['VLLM_DECODE_BS_BUCKET_STEP']),
    )
    fd['decode_block_ramp_graphs'] = 1 + int(
        math.log(
            fd['VLLM_DECODE_BLOCK_BUCKET_STEP'] /
            fd['VLLM_DECODE_BLOCK_BUCKET_MIN'],
            2,
        ))
    fd['decode_block_step_graphs'] = max(
        0,
        int(1 + (fd['est_hpu_blocks'] - fd['VLLM_DECODE_BLOCK_BUCKET_STEP']) /
            fd['VLLM_DECODE_BLOCK_BUCKET_STEP']),
    )
    fd['num_decode_graphs'] = (
        (fd['decode_bs_ramp_graphs'] + fd['decode_bs_step_graphs']) *
        (fd['decode_block_ramp_graphs'] + fd['decode_block_step_graphs']))
    fd['prompt_bs_ramp_graphs'] = 1 + int(
        math.log(
            min(fd['max_num_prefill_seqs'], fd['VLLM_PROMPT_BS_BUCKET_STEP']) /
            fd['VLLM_PROMPT_BS_BUCKET_MIN'],
            2,
        ))
    fd['prompt_bs_step_graphs'] = max(
        0,
        int(1 +
            (fd['max_num_prefill_seqs'] - fd['VLLM_PROMPT_BS_BUCKET_STEP']) /
            fd['VLLM_PROMPT_BS_BUCKET_STEP']),
    )
    fd['prompt_seq_ramp_graphs'] = 1 + int(
        math.log(
            fd['VLLM_PROMPT_SEQ_BUCKET_STEP'] /
            fd['VLLM_PROMPT_SEQ_BUCKET_MIN'],
            2,
        ))
    fd['prompt_seq_step_graphs'] = int(
        1 + (fd['max_model_len'] - fd['VLLM_PROMPT_SEQ_BUCKET_STEP']) /
        fd['VLLM_PROMPT_SEQ_BUCKET_STEP'])
    fd['est_num_prompt_graphs'] = (
        (fd['prompt_bs_ramp_graphs'] + fd['prompt_bs_step_graphs']) *
        (fd['prompt_seq_ramp_graphs'] + fd['prompt_seq_step_graphs']) / 2)
    fd['est_graph_prompt_ratio'] = math.ceil(
        fd['est_num_prompt_graphs'] /
        (fd['est_num_prompt_graphs'] + fd['num_decode_graphs']) * 100) / 100
    print(f"Estimated Prompt graphs {fd['est_num_prompt_graphs']:.0f} and "
          f"Decode graphs {fd['num_decode_graphs']}")
    fd['VLLM_GRAPH_PROMPT_RATIO'] = math.ceil(
        min(max(fd['est_graph_prompt_ratio'], 0.1), 0.9) * 10) / 10
    fd['decode_graph_target_GB'] = math.ceil(
        fd['num_decode_graphs'] * fd['approx_mem_per_graph_MB'] / 1024 *
        10) / 10
    fd['est_graph_reserve_mem'] = math.ceil(
        fd['decode_graph_target_GB'] /
        (fd['usable_mem'] * fd['gpu_memory_utilization'] *
         (1 - fd['VLLM_GRAPH_PROMPT_RATIO'])) * 100) / 100
    fd['VLLM_GRAPH_RESERVED_MEM'] = min(max(fd['est_graph_reserve_mem'], 0.01),
                                        0.5)
    fd['kv_cache_mem'] = (fd['usable_mem'] * fd['gpu_memory_utilization'] *
                          (1 - fd['VLLM_GRAPH_RESERVED_MEM']))

    if fd.get('max_num_seqs') is None:
        fd['max_num_seqs'] = (fd['tensor_parallel_size'] * fd['kv_cache_mem'] /
                              fd['kv_cache_per_seq'])
        if dtype == 'fp8':
            fd['max_num_seqs'] = (max(
                1,
                math.floor(
                    fd['max_num_seqs'] / fd['VLLM_DECODE_BS_BUCKET_STEP']),
            ) * fd['VLLM_DECODE_BS_BUCKET_STEP'])
        else:
            fd['max_num_seqs'] = (math.ceil(
                fd['max_num_seqs'] / fd['VLLM_DECODE_BS_BUCKET_STEP']) *
                                  fd['VLLM_DECODE_BS_BUCKET_STEP'])

        if fd['max_num_seqs'] < 1:
            raise ValueError(
                "Not enough memory for kv cache increase tensor_parallel_size "
                "or reduce max_model_len or increase bucket step")
    else:
        fd['max_num_seqs'] = max(1, fd['max_num_seqs'])

    fd['VLLM_DECODE_BLOCK_BUCKET_MAX'] = max(
        128, math.ceil((fd['max_num_seqs'] * fd['max_model_len']) / 128))
    fd['VLLM_PROMPT_SEQ_BUCKET_MAX'] = fd['max_model_len']

    # Create our output list
    with open('varlist_output.txt') as ovp_file:
        for line in ovp_file:
            param = line.strip()
            output_dict[param] = fd[param]

    # Append user updatable list
    with open('varlist_userupd.txt') as ovp_file:
        for line in ovp_file:
            param = line.strip()
            if os.environ.get(param) is not None:
                output_dict[param] = fd[param]

    return output_dict


def get_model_from_csv(file_path):
    # Read settings CSV and return dict
    dataframe_csv = pd.read_csv(file_path)
    filtered_row = dataframe_csv.loc[dataframe_csv['model'] ==
                                     os.environ['MODEL']]

    if filtered_row.empty:
        raise ValueError(
            f"No matching rows found for model '{os.environ['MODEL']}' "
            f"in {file_path}")

    # CSV should not have more than 1 row for each model.
    # But just in case, return the first
    try:
        filtered_dict = filtered_row.to_dict(orient='records')[0]
    except Exception as err:
        raise ValueError(
            "Unsupported model or model not defined! Exiting.") from err

    return filtered_dict


def overwrite_params(dict_before_updates):
    # Overwrite default values with user provided ones before auto_calc
    with open('varlist_userupd.txt') as ovp_file:
        for line in ovp_file:
            param = line.strip()
            if os.environ.get(param) is not None:
                try:
                    dict_before_updates[param] = eval(os.environ[param])
                except Exception:
                    dict_before_updates[param] = os.environ[param]

                print(f"Adding or updating {param} \
                        to {dict_before_updates[param]}")

    return dict_before_updates


def write_dict_to_file(fd, file):
    with open(file, 'w') as file_obj:
        for key, value in fd.items():
            file_obj.write(f"export {key}={value}\n")


def main():
    global hpu_mem, HPU_determined, dtype, output_dict

    # CONSTANTS
    hpu_mem = {'GAUDI2': 96, 'GAUDI3': 128}
    # TODO: Remove this hardcoded value in the future
    dtype = "bfloat16"

    # PRECHECKS
    if os.getenv('MODEL') is None:
        print('Error no model. Provide model name in env var "MODEL"')
        exit(-1)

    # Output vars
    file_input_csv = 'settings_vllm.csv'
    file_output_vars = 'server_vars.txt'
    output_dict = {}

    # Get HPU model and filter row by HPU again
    HPU_determined = get_device_model()

    # Read settings csv into a dataframe
    try:
        fd = get_model_from_csv(file_input_csv)
    except ValueError as e:
        print("Error:", e)
        exit(-1)

    # Use a single if statement for MAX_MODEL_LEN
    if (os.getenv('MAX_MODEL_LEN') is not None
            and int(os.environ['MAX_MODEL_LEN']) > fd['limit_model_length']):
        print(f"Supplied max_model_length {os.environ['MAX_MODEL_LEN']} "
              "cannot be higher than the permissible value "
              f"{str(fd['limit_model_length'])} for this model.")
        exit(-1)

    # Overwrite params then perform autocalc
    fd = overwrite_params(fd)
    try:
        output_dict = vllm_auto_calc(fd)
    except ValueError as e:
        print("Error:", e)
        exit(-1)

    # Write to a text file
    write_dict_to_file(output_dict, file_output_vars)


if __name__ == '__main__':
    main()
