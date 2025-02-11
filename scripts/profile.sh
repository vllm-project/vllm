#!/bin/bash
cur_path=$(pwd)
HABANA_PROFILE_WRITE_HLTV=1 HABANA_PROFILE=profile_api \
VLLM_PT_PROFILE=decode_128_2048_t \
python ${cur_path}/run_example_tp.py