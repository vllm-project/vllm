#!/bin/bash -ex

HOSTFILE=""
MASTER_ADDR=""
MASTER_PORT=""

usage() {
    echo "Usage: $0 -h <hostfile> -a <master_address> -p <master_port> <python_command>"
    exit 1
}

while getopts "h:a:p:" opt; do
    case "$opt" in
        h) HOSTFILE=$OPTARG ;;
        a) MASTER_ADDR=$OPTARG ;;
        p) MASTER_PORT=$OPTARG ;;
        *) usage ;;
    esac
done

shift $((OPTIND - 1))

if [ -z "$HOSTFILE" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
    echo "Error: Missing required arguments."
    usage
fi

echo "Using hostfile: $HOSTFILE"
echo "Using address: $MASTER_ADDR"
echo "Using port: $MASTER_PORT"
echo "Python command:"
echo "$@"

# Use mpirun to trigger inference on head/worker nodes

/opt/amazon/openmpi/bin/mpirun \
  --mca mtl ^ofi --mca btl tcp,self --bind-to none \
  -np 2 \
  --hostfile "$HOSTFILE"\
  --prefix /opt/amazon/openmpi \
  -x FI_PROVIDER=efa \
  -x FI_EFA_USE_DEVICE_RDMA=1 \
  -x FI_EFA_FORK_SAFE=1 \
  -x PATH=/opt/amazon/openmpi/bin:$PATH \
  -x PYTHONPATH=$PYTHONPATH \
  -x LD_LIBRARY_PATH=/opt/aws/neuron/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH \
  -x MASTER_ADDR="$MASTER_ADDR" -x MASTER_PORT="$MASTER_PORT" \
  "$@"