set -x
#export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/
export MODEL_PATH=/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/


if [ -z "$1" ]; then
    echo "please input the dp size per node, for example, 16dp on 2 node, run the xxx.sh 8"
    echo "run with default mode n=8"
    NUM_DECODE=8
else
    NUM_DECODE=$1
fi

DECODE_IPS=("10.239.129.81" "10.239.129.165" "10.239.129.67" "10.239.129.21")
BASE_PORT=8200
DECODE_ARGS=""

for ((i=0; i<$NUM_DECODE; i++)); do
    PORT=$((BASE_PORT + i))
    for IP in "${DECODE_IPS[@]}"; do
        DECODE_ARGS="$DECODE_ARGS ${IP}:${PORT}"
    done
done

python3 ./examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill 10.239.129.9:8100 \
    --decode $DECODE_ARGS \
    --port 8868
