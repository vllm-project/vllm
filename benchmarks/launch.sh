#!/bin/bash

usage() {
       echo "Options:                                                                                      "
       echo "    -h|--help                 help for launch.sh                                              "
       echo "    --server                  server backend e.g. tgi                                         "
       echo "    -p|--port                 port for server                                                 "
       echo "    -m|--model                modelname e.g. facebook/opt-125m                                "
       echo "    --max-batch-total-tokens  max-batch-total-tokens                                          "
       echo "    --max-input-length        max-input-length                                                "
       echo "    --max-total-tokens        max-total-tokens                                                "
       echo "    --max-best-of             max-best-of                                                     "
       echo "    --max-concurrent-requests max-concurrent-requests                                         "
       echo "    --tgi-image               image for tgi server                                            "
       echo "    --sharded                 shard or not, e.g. false                                        "
       echo "Usage:                                                                                        "
       echo "    sh launch.sh --server tgi --model facebook/opt-125m --max-batch-total-tokens 10000 -p 8000"
       echo "    sh launch.sh --server tgi -m facebook/opt-125m --max-batch-total-tokens 10000 -p 8000     "
       exit 0
}

# default value
SERVER=tgi
MODEL=facebook/opt-125m
PORT=8000
SHM_SIZE=1G
MAX_BEST_OF=5
MAX_BATCH_TOTAL_TOKENS=10000
MAX_INPUT_LENGTH=1024
MAX_TOTAL_TOKENS=2048
MAX_CONCURRENT_REQUESTS=5000
TGI_IMAGE=ghcr.io/huggingface/text-generation-inference:1.4.0
SHARDED=false

ARGS=`getopt -o hm:p: --long server:,port:,model:,max-batch-total-tokens:,shm-size:,\
max-input-length:,max-total-tokens:,max-best-of:,max-concurrent-requests:,tgi-image:\
sharded:,help -n "$0" -- "$@"`

if [ $? != 0 ]; then
       echo "Terminating..."
       exit 1
fi

eval set -- "${ARGS}"

while true
do
       case "$1" in
              --server)                  SERVER=$2                  ; shift 2;;
              -p|--port)                 PORT=$2                    ; shift 2;;
              -m|--model)                MODEL=$2                   ; shift 2;;
              --max-batch-total-tokens)  MAX_BATCH_TOTAL_TOKENS=$2  ; shift 2;;
              --max-input-length)        MAX_INPUT_LENGTH=$2        ; shift 2;;
              --max-total-tokens)        MAX_TOTAL_TOKENS=$2        ; shift 2;;
              --max-best-of)             MAX_BEST_OF=$2             ; shift 2;;
              --max-concurrent-requests) MAX_CONCURRENT_REQUESTS=$2 ; shift 2;;
              --shm-size)                SHM_SIZE=$2                ; shift 2;;
              --tgi-image)               TGI_IMAGE=$2               ; shift 2;;
              --sharded)                 SHARDED=$2                 ; shift 2;;
              -h|--help)                 usage                      ; shift  ;;
              --)                        shift                      ; break  ;;
              *)                         echo "Internal error!"     ; exit 1 ;;
       esac
done

case "$SERVER" in
       tgi)
              echo "tgi"
              set -x
               docker run -e HF_TOKEN=$HF_TOKEN --gpus all --shm-size $SHM_SIZE -p $PORT:80 \
                       -v $PWD/data:/data \
                       $TGI_IMAGE \
                       --model-id $MODEL \
                       --sharded $SHARDED  \
                       --max-input-length $MAX_INPUT_LENGTH \
                       --max-total-tokens $MAX_TOTAL_TOKENS \
                       --max-best-of $MAX_BEST_OF \
                       --max-concurrent-requests $MAX_CONCURRENT_REQUESTS \
                       --max-batch-total-tokens $MAX_BATCH_TOTAL_TOKENS
       ;;
       *)
               echo "Unknown server, exit."
               usage
       ;;
esac

