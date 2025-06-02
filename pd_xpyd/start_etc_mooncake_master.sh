export PATH=$PATH:/usr/local/lib/python3.10/dist-packages/mooncake/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/mooncake

pkill -f mooncake_master
pkill -f etcd
sleep 5s

etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379  >etcd.log 2>&1 &
mooncake_master -enable_gc true -max_threads 64 -port 50001 --v=1 >mooncake_master.log 2>&1 &
 # -v 2

