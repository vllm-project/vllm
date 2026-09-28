#!/usr/bin/env bash
# Select the network userspace stack for this container, then exec the command.
#   NET_STACK=auto (default): use the AWS EFA stack if an EFA NIC is visible
#   NET_STACK=efa:            always use the AWS EFA stack from /opt/amazon
#   NET_STACK=system:         use the system rdma-core / libfabric (IB, RoCE)
# The AWS libraries only work as a set (libfabric needs the AWS libefa and
# libibverbs), so they are switched together.
NET_STACK="${NET_STACK:-auto}"
if [ "${NET_STACK}" = auto ]; then
    NET_STACK=system
    for d in /sys/class/infiniband/*/device/driver; do
        if [ "$(basename "$(readlink -f "$d")")" = efa ]; then
            NET_STACK=efa
            break
        fi
    done
fi
if [ "${NET_STACK}" = efa ] && [ -d /opt/amazon/efa/lib ]; then
    export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="/opt/amazon/efa/bin:${PATH}"
fi
export NET_STACK
exec "$@"
