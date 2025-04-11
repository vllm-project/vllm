export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379  >etcd.log 2>&1 &
mooncake_master -enable_gc true -port 50001 # -v 2

