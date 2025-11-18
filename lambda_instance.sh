#!/bin/bash
# Lambda Labs Instance Helper Script
# Instance ID: 0b84a041d4544e72ad453da7bf2c5b38

API_KEY="secret_sheikh-abdur-rahim_6f5449ac2d1b4d55b62737b6d8d26068.8olMhij6fSWEj1SybGGJPAu58K5rrZWg"
INSTANCE_ID="0b84a041d4544e72ad453da7bf2c5b38"

# Function to check instance status
check_status() {
    echo "Checking instance status..."
    curl -s -u "$API_KEY:" https://cloud.lambdalabs.com/api/v1/instances | jq '.data[0]'
}

# Function to get instance IP
get_ip() {
    IP=$(curl -s -u "$API_KEY:" https://cloud.lambdalabs.com/api/v1/instances | jq -r '.data[0].ip // empty')
    if [ -z "$IP" ]; then
        echo "Instance is still booting or IP not yet assigned"
        return 1
    else
        echo "Instance IP: $IP"
        echo "SSH command: ssh ubuntu@$IP"
        return 0
    fi
}

# Function to terminate instance
terminate() {
    echo "Terminating instance $INSTANCE_ID..."
    curl -u "$API_KEY:" \
      https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
      -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}" \
      -H "Content-Type: application/json" | jq .
}

# Main menu
case "${1:-status}" in
    status)
        check_status
        ;;
    ip)
        get_ip
        ;;
    ssh)
        IP=$(curl -s -u "$API_KEY:" https://cloud.lambdalabs.com/api/v1/instances | jq -r '.data[0].ip // empty')
        if [ -n "$IP" ]; then
            echo "Connecting to $IP..."
            ssh ubuntu@$IP
        else
            echo "Instance IP not available yet. Try again in a moment."
        fi
        ;;
    terminate)
        terminate
        ;;
    *)
        echo "Usage: $0 {status|ip|ssh|terminate}"
        echo "  status    - Check instance status"
        echo "  ip        - Get instance IP address"
        echo "  ssh       - SSH into the instance"
        echo "  terminate - Terminate the instance"
        exit 1
        ;;
esac
