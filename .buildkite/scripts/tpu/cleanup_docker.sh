#!/bin/bash

set -euo pipefail

docker_root=$(docker info -f '{{.DockerRootDir}}')
if [ -z "$docker_root" ]; then
  echo "Failed to determine Docker root directory."
  exit 1
fi
echo "Docker root directory: $docker_root"
# Check disk usage of the filesystem where Docker's root directory is located
disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
# Define the threshold
threshold=70
if [ "$disk_usage" -gt "$threshold" ]; then
  echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
  # Remove dangling images (those that are not tagged and not used by any container)
  docker image prune -f
  # Remove unused volumes / force the system prune for old images as well.
  docker volume prune -f && docker system prune --force --filter "until=24h" --all
  echo "Docker images and volumes cleanup completed."
else
  echo "Disk usage is below $threshold%. No cleanup needed."
fi
