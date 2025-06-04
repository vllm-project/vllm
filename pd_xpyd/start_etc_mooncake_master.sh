#!/bin/bash

export PATH=$PATH:/usr/local/lib/python3.10/dist-packages/mooncake/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/mooncake

pkill -f mooncake_master
pkill -f etcd
sleep 5s

has_arg() {
  local keyword=$1
  shift
  for arg in "$@"; do
    if [ "$arg" == "$keyword" ]; then
      return 0
    fi
  done
  return 1
}

DEBUG_MODE=0

if has_arg debug "$@"; then
  echo "Debug mode enabled"
  DEBUG_MODE=1
fi

etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://localhost:2379 \
     >etcd.log 2>&1 &

if [ "$DEBUG_MODE" == "1" ]; then
  mooncake_master -max_threads 64 -port 50001 --v=1 >mooncake_master.log 2>&1 &
else
  mooncake_master -enable_gc true -max_threads 64 -port 50001 --v=1 >mooncake_master.log 2>&1 &
fi

