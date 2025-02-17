#!/bin/bash
cur_path=$(pwd)
HABANA_PROFILE_WRITE_HLTV=1 HABANA_PROFILE=1
VLLM_PT_PROFILE=decode_128_1024_t \
HABANA_PROF_CONFIG=${cur_path}/profile_api_trace_analyzer.json \
python ${cur_path}/run_example_tp.py