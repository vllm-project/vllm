#!/bin/bash

model=/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B
model_short=$(basename $model)
#replace all '.' with '-'
model_short=${model_short//./-}
log_file=$(mktemp /tmp/_benchmark_${model_short}_XXXXXXX.log)

# Generate an empty result file.
# This way in case of any crash it will be treated by jenkins as failure
if [[ -n "$TEST_RESULTS_DIR" ]]; then
    mkdir -p ${TEST_RESULTS_DIR}
    LOG_PATH=$(mktemp ${TEST_RESULTS_DIR}/benchmark_${model_short}_XXXXXX.xml)
fi

throughput_threshold=999999 
if [[ "${PERF_THRESHOLD}" ]]; then
    throughput_threshold=${PERF_THRESHOLD}
fi
warmup_threshold=1 
if [[ "${WARMUP_THRESHOLD}" ]]; then
    warmup_threshold=${WARMUP_THRESHOLD}
fi

# Get the directory of the current script
script_dir=$(dirname "$(readlink -f "$0")")

start=`date +%s`
python  $script_dir/../../benchmarks/benchmark_throughput.py \
    --model $model \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset /mnt/weka/data/pytorch/llama2/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --use-padding-aware-scheduling |& tee $log_file
end=`date +%s`
runtime=$((end-start))
printf " -------------- \nBenchmark took: %2d:%02d\n\n" $((runtime/60)) $((runtime%60)) 

throughput=$(grep -oP 'Throughput: [0-9.]+ requests/s, \K[0-9]+' $log_file)
warmup=$(grep -oP 'Warmup finished in +\K[0-9:]+' $log_file)

warmup_status="FAILED"
warmup_fail=1
if [[ "$warmup" ]]; then
    if ((warmup <= warmup_threshold)); then
        warmup_status="PASSED"
        warmup_fail=0
    fi
fi
echo "=== $warmup_status warmup MODEL: ${model_short}  ($warmup <= $warmup_threshold) ==="
throughput_status="FAILED"
throughput_fail=1
if [[ "$throughput" ]]; then
    if ((throughput >= throughput_threshold)); then
        throughput_status="PASSED"
        throughput_fail=0
    fi
fi
echo "=== $throughput_status throughput MODEL: ${model_short}  ($throughput >= $throughput_threshold) ==="

if [[ -n "$TEST_RESULTS_DIR" ]]; then
    # Store full benchmark log
    chmod +r $log_file
    mv $log_file ${TEST_RESULTS_DIR}/

    # Report results for jenkins
    cat <<EOF > ${LOG_PATH}
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="benchmark" errors="0" failures="$((throughput_fail + warmup_fail))" skipped="0" tests="2" time="$runtime">
<testcase classname=".jenkins.benchmark.${model_short}-bf16" name="${model_short}-bf16-throughput" time="$runtime">
<properties>
<property name="throughput" value="$throughput"/>
<property name="throughput threshold" value="$throughput_threshold"/>
</properties>
EOF
    if [[ "$throughput_fail" -eq 1 ]]; then
        cat <<EOF >> ${LOG_PATH}
<failure message="Throughput did not meet the threshold  ($throughput &lt; $throughput_threshold)"></failure>
EOF
    fi
 cat <<EOF >> ${LOG_PATH}
</testcase>
<testcase classname=".jenkins.benchmark.${model_short}-bf16" name="${model_short}-bf16-warmup" time="$warmup">
<properties>
<property name="warmup time" value="$warmup"/>
<property name="warmup threshold" value="$warmup_threshold"/>
</properties>
EOF
    if [[ "$warmup_fail" -eq 1 ]]; then
        cat <<EOF >> ${LOG_PATH}
<failure message="Warmup did not meet the threshold ($warmup &gt; $warmup_threshold)"></failure>
EOF
    fi
    cat <<EOF >> ${LOG_PATH}
</testcase>
</testsuite>
</testsuites>
EOF

fi
