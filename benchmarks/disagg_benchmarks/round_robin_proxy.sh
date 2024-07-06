#!/bin/bash

# Define the ports to forward to
PORTS=(8100 8200)
NUM_PORTS=${#PORTS[@]}
CURRENT=0

# Function to handle the round-robin logic
get_next_port() {
  NEXT_PORT=${PORTS[$CURRENT]}
  CURRENT=$(( (CURRENT + 1) % NUM_PORTS ))
  echo $NEXT_PORT
}

# Start the proxy
while true; do
  NEXT_PORT=$(get_next_port)
  echo "Forwarding to port $NEXT_PORT"
  socat TCP4-LISTEN:8000,reuseaddr,fork TCP4:localhost:$NEXT_PORT
done