#!/bin/bash
HOST="localhost"
PORT=8006

python examples/online_serving/elastic_ep/scale.py --host $HOST --port $PORT --new-dp-size 4
