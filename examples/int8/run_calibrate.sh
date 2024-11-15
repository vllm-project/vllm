#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
datasets_path=/home/datasets/
work_dir=./work_dir/
datasets_name="ceval_val_cmcc ceval cmmlu cmb medmcqa medqa mmlu"
csv_name=LLaMA-Factory/evaluation/
log_dir=./cali_log/
for i in $datasets_name;
do
    if [ "$i" == "ceval_val_cmcc" ]; then
        calib_dataset_path=${datasets_path}
    else
        calib_dataset_path=${datasets_path}${csv_name}$i/
    fi
    save_dir=${work_dir}$i/pth/
    [ ! -d ${save_dir} ] && mkdir ${save_dir}
    [ ! -d ${log_dir} ] && mkdir ${log_dir}
    log=${log_dir}llama3-8b-datasets_$i.log
    echo "i=$i, calib_dataset_path=${calib_dataset_path}, save_dir=${save_dir}, log=${log}"
    python calibrate.py /home/model_weights/Llama3-Chinese-8B-Instruct/ \
            --calib_dataset $i \
            --dataset_path  ${calib_dataset_path} \
            --work_dir ${save_dir} \
            --device cuda\
            --calib_samples 128 \
            --calib_seqlen 2048  2>&1|tee ${log} 
    log=${log_dir}llama3-8b-datasets_${i}_json.log
    save_dir_path=${work_dir}$i/
    python export_kv_params.py \
        --work_dir ${save_dir} \
        --kv_params_dir ${save_dir_path} \
        --quant_group 8  2>&1|tee ${log} 
done

