#!/bin/bash

apt-get update
curl -fsSL https://keys.openpgp.org/vks/v1/by-fingerprint/32A37959C2FA5C3C99EFBC32A79206696452D198 | sudo gpg --dearmor -o /usr/share/keyrings/buildkite-agent-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/buildkite-agent-archive-keyring.gpg] https://apt.buildkite.com/buildkite-agent stable main" | sudo tee /etc/apt/sources.list.d/buildkite-agent.list
apt-get update
apt-get install -y buildkite-agent

sudo usermod -a -G docker buildkite-agent
sudo -u buildkite-agent gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

sudo sed -i "s/xxx/d31d2b6b82f2bf40556f4da89489a7bc8bf60f401d5c60e504/g" /etc/buildkite-agent/buildkite-agent.cfg
sudo sed -i "s/%hostname-%spawn/tpu-1/g" /etc/buildkite-agent/buildkite-agent.cfg
sudo sh -c 'echo "tags=\"queue=tpu\"" >> /etc/buildkite-agent/buildkite-agent.cfg'
export HF_TOKEN="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"
systemctl enable buildkite-agent
systemctl start buildkite-agent