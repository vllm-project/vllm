#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/dp_d_env.sh

export VLLM_EP_SIZE=16
export MOONCAKE_CONFIG_PATH=./pd_xpyd/2p2d_mooncake_d1.json

unset VLLM_DP_SIZE
unset VLLM_USE_V1
unset VLLM_DP_MASTER_IP
unset VLLM_DP_MASTER_PORT

ray start --address='10.239.129.81:8826'


