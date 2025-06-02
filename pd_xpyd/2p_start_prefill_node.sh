#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_${1:-g12}.json

source "$BASH_DIR"/dp_p_env.sh

ray start --address="${2:-10.239.129.9:6886}"

