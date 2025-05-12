#set -x

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH=./pd_xpyd/2p4d_mooncake_d3.json 

source ./pd_xpyd/dp_p_env.sh

ray start --address='10.239.129.67:6888'

