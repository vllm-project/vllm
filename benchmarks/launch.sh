#!/bin/bash

usage() {
       echo "Options:                                                                          "
       echo "    -h|--help   help for launch.sh                                                "
       echo "    --server    server backend e.g. tgi                                           "
       echo "    -p|--port   port for server                                                   "
       echo "    -m|--model  modelname e.g. facebook/opt-125m                                  "
       echo "    -t|--tokens max-batch-total-tokens                                            "
       echo "Usage:                                                                            "
       echo "    sh launch.sh --server tgi --model facebook/opt-125m --tokens 10000 --port 8000"
       echo "    sh launch.sh --server tgi -m facebook/opt-125m -t 10000 -p 8000               "
       exit 0
}

ARGS=`getopt -o hm:p:t: --long server:,port:,model:,tokens:,help -n "$0" -- "$@"`

if [ $? != 0 ]; then
       echo "Terminating..."
       exit 1
fi

eval set -- "${ARGS}"

while true
do
       case "$1" in
               --server)    export SERVER=$2       ; shift 2;;
               -p|--port)   export PORT=$2         ; shift 2;;
               -m|--model)  export MODEL=$2        ; shift 2;;
               -t|--tokens) export TOKENS=$2       ; shift 2;;
               -h|--help)   usage                  ; shift  ;;
               --)          shift                  ; break  ;;
               *)           echo "Internal error!" ; exit 1 ;;
       esac
done

case "$SERVER" in
       tgi)
               echo "tgi"
               docker run -e HF_TOKEN=$HF_TOKEN --gpus all --shm-size 1g -p $PORT:80 \
                       -v $PWD/data:/data \
                       ghcr.io/huggingface/text-generation-inference:1.4.0 \
                       --model-id $MODEL \
                       --sharded false  \
                       --max-input-length 1024 \
                       --max-total-tokens 2048 \
                       --max-best-of 5 \
                       --max-concurrent-requests 5000 \
                       --max-batch-total-tokens $TOKENS
       ;;
       *)
               echo "Unknown server, exit."
               usage
       ;;
esac

