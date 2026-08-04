#!/usr/bin/env bash
# Install the GPU monitor as a cron job on the CI host machine.
# Run this once during initial cluster setup.
#
# Usage: sudo bash setup-gpu-monitor.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/gpu-monitor.sh"
LOG_FILE="/var/log/gpu-monitor.log"
CRON_INTERVAL="*/10"  # Every 10 minutes

if [[ ! -f "$MONITOR_SCRIPT" ]]; then
    echo "ERROR: gpu-monitor.sh not found at $MONITOR_SCRIPT"
    exit 1
fi

chmod +x "$MONITOR_SCRIPT"

# Create log file with rotation
touch "$LOG_FILE"

# Set up logrotate
cat > /etc/logrotate.d/gpu-monitor <<EOF
${LOG_FILE} {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF

# Install cron job (idempotent: removes old entry first)
CRON_CMD="${CRON_INTERVAL} * * * * ${MONITOR_SCRIPT} >> ${LOG_FILE} 2>&1"
(crontab -l 2>/dev/null | grep -v "gpu-monitor.sh" || true; echo "$CRON_CMD") | crontab -

echo "GPU monitor installed:"
echo "  Script: $MONITOR_SCRIPT"
echo "  Log:    $LOG_FILE"
echo "  Cron:   $CRON_CMD"
echo ""
echo "View logs:    tail -f $LOG_FILE"
echo "Remove cron:  crontab -l | grep -v gpu-monitor.sh | crontab -"
