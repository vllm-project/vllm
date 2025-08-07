#!/bin/bash
pip install vllm -U
pip install transformers -U
export PATH=/home/jeeves/.local/bin:$PATH
cd src/
model_path=$1
model_name=$2
for task in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
    python eval_yi_200k.py --task ${task} --model_path ${model_path} --output_dir /data/checkpoints/ --model_name ${model_name}
done