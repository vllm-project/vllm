#!/bin/bash
cur_path=$(pwd)
HABANA_PROFILE_WRITE_HLTV=1 HABANA_PROFILE=1 \
HABANA_PROF_CONFIG=${cur_path}/profile_api_trace_analyzer.json \
VLLM_PT_PROFILE=decode_16_4000_t \
python ${cur_path}/run_hpu_example.py