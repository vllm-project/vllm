#!/bin/bash

set -e  # Exit on error

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <device> <mount_dir> <owner:group>"
  echo "Example: $0 /dev/nvme0n1p4 /holly hcasalet:microfastpath-PG"
  exit 1
fi

DEVICE=$1
MOUNT_DIR=$2
OWNER_GROUP=$3

# Make filesystem
echo "📀 Creating ext4 filesystem on $DEVICE ..."
sudo mkfs.ext4 -F "$DEVICE"

# Create and mount directory
echo "📁 Creating mount point at $MOUNT_DIR ..."
sudo mkdir -p "$MOUNT_DIR"
echo "🔗 Mounting $DEVICE to $MOUNT_DIR ..."
sudo mount "$DEVICE" "$MOUNT_DIR"


# Set ownership
echo "🔒 Changing ownership to $OWNER_GROUP ..."
sudo chown -R "$OWNER_GROUP" "$MOUNT_DIR"

echo "✅ Done creating file system. $DEVICE mounted at $MOUNT_DIR with ownership set to $OWNER_GROUP."

# Install prerequisites
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release

# Add Docker GPG key safely
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Enable and start Docker
sudo systemctl enable --now docker

# Add current user to docker group
REAL_USER=$(logname)
sudo usermod -aG docker $REAL_USER
echo "✅ Added $REAL_USER to docker group. Please log out and back in for it to take effect."

# Install Docker Compose (v2.29.1)
sudo curl -L "https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker --version
docker-compose --version

# Move docker to partition with higher compacity
echo "🛠️ Moving Docker data directory to $MOUNT_DIR/docker ..."
if [ ! -d $MOUNT_DIR ]; then
  echo "❌ Directory /holly does not exist. Please create or mount it first."
  exit 1
fi

sudo systemctl stop docker
sudo mkdir -p $MOUNT_DIR/docker
sudo rsync -aP /var/lib/docker/ $MOUNT_DIR/docker/
sudo mv /var/lib/docker /var/lib/docker.bak
sudo ln -s $MOUNT_DIR/docker /var/lib/docker
sudo systemctl start docker
sudo rm -rf /var/lib/docker.bak

sudo apt-get update
sudo apt-get install expect sysstat iotop dstat procps htop jq -y

# set up Python env
sudo apt update
sudo apt install python3.10-venv

# gen key for ssh
ssh-keygen -t ed25519 -C "hcasalet@ucsc.edu"
cat ~/.ssh/id_ed25519.pub
