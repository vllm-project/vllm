#!/bin/bash
save_dir=/lustre/fsw/portfolios/hw/users/sshrestha/infinitebench_data
mkdir ${save_dir}
for file in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
    wget -c https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/${file}.jsonl?download=true -O ${save_dir}/${file}.jsonl
done