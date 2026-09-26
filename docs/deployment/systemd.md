# Using systemd

`vllm serve` can run as a systemd service. Because loading a model takes anywhere from seconds
to minutes, use a `Type=notify` unit: the server tells systemd when it is actually ready to
answer requests, so `systemctl start` blocks until then and units that declare `After=` /
`Requires=` on it only start once the model can be queried.

```ini
[Unit]
Description=vLLM OpenAI-compatible server
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
# The API server may run in a child process (e.g. with --api-server-count),
# so accept the notification from any process of the unit.
NotifyAccess=all
# Loading a large model can exceed the default 90 s.
TimeoutStartSec=30min
ExecStart=/opt/vllm/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

vLLM sends `READY=1` on the socket systemd passes in `NOTIFY_SOCKET` once the HTTP server is
bound and serving; the log line `Notified systemd that the server is ready` confirms it. With
`Type=simple` or `Type=exec`, or outside systemd, nothing changes: the notification is skipped.

The same works for transient units, which is convenient to start a model and wait for it from
a script:

```bash
systemd-run --user --property=Type=notify --property=NotifyAccess=all \
  --property=TimeoutStartSec=30min --unit=vllm-llama \
  vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
# returns when the server answers, then:
curl -s localhost:8000/health
```
