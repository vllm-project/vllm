#set +x
export MODEL_PATH=/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

if [ -z "$1" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D"
    echo "run with default mode P=1, D=2, TP size=1, false"
    P_INSTANCE_NUMBER=1
    D_INSTANCE_NUMBER=2
    NUM_DECODE=8
    FIRST_TOKEN_FROM_P=false
else
    P_INSTANCE_NUMBER=$1
fi

if [ -z "$2" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D"
    echo "run with P=$P_INSTANCE_NUMBER, D=2, TP size=1, false"
    D_INSTANCE_NUMBER=2
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
    FIRST_TOKEN_FROM_P=false
else
    D_INSTANCE_NUMBER=$2
fi

if [ -z "$3" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=1, false"
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
    FIRST_TOKEN_FROM_P=false
else
    TP_SIZE=$3
    NUM_DECODE=$((8 / TP_SIZE))
fi

if [ -z "$4" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=$TP_SIZE, false"
    FIRST_TOKEN_FROM_P=false
else
    FIRST_TOKEN_FROM_P=$4
fi

DECODE_IPS=("10.239.129.81" "10.239.129.165" "10.239.129.67" "10.239.129.21")
DBASE_PORT=8200
DECODE_ARGS=""

for ((i=0; i<D_INSTANCE_NUMBER; i++)); do
    IP=${DECODE_IPS[$i]}
    for ((j=0; j<NUM_DECODE; j++)); do
        PORT=$((DBASE_PORT + j))
        DECODE_ARGS="$DECODE_ARGS ${IP}:${PORT}"
    done
done


PREFILL_IPS=("10.239.129.9" "10.239.129.24" "10.239.129.67" "10.239.129.21")
PBASE_PORT=8100
PREFILL_ARGS=""

PORT=$PBASE_PORT
for ((i=0; i<P_INSTANCE_NUMBER; i++)); do
    IP=${PREFILL_IPS[$i]}
    PREFILL_ARGS="$PREFILL_ARGS ${IP}:${PORT}"
done

CMD="python3 ./examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill $PREFILL_ARGS \
    --decode $DECODE_ARGS \
    --port 8868"

if [ "$FIRST_TOKEN_FROM_P" = "true" ]; then
    CMD="$CMD --generator_on_p_node"
fi

echo "Running: $CMD"
eval $CMD

