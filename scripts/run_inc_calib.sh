DEFAULT_MODEL_PATH="/mnt/disk3/DeepSeek-R1-G2-INC"
FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

QUANT_CONFIG_FILE="scripts/quant_configs/inc_measure_with_fp8kv_config.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="prepare.pile.512.${timestamp}.log"

# remove ./scripts/nc_workspace_measure_kvache if needed
if [ -e ./scripts/nc_workspace_measure_kvache ]; then
    echo "The directory ./scripts/nc_workspace_measure_kvache already exists, removing it..."
    rm -rf ./scripts/nc_workspace_measure_kvache
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"


PT_HPU_LAZY_MODE=1 \
VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
VLLM_DECODE_BS_BUCKET_MIN=1 \
VLLM_DECODE_BS_BUCKET_MAX=1 \
VLLM_REQUANT_FP8_INC=1 \
VLLM_MOE_N_SLICE=1 \
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python scripts/run_example_tp.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --nprompts 512 \
    --max_model_len 2048 \
    --dataset pile 2>&1 | tee $LOG_FILE
