#!/bin/bash
exec > eval_vllm_sparse_llama_infinitebench.txt 2>&1

# Run all tasks sequentially
# python eval_vllm.py --task passkey --ngpu 8
# python eval_vllm.py --task number_string --ngpu 8
# python eval_vllm.py --task kv_retrieval --ngpu 8

#!/bin/bash
# for task in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
for task in code_debug code_run longbook_choice_eng longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find ; do    
    python eval_vllm.py --task ${task} --ngpu 8
done