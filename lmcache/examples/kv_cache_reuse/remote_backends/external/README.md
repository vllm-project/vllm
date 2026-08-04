# External Connector for LMCache

LMCache supports custom external storage backends via Python modules. This connector type allows integrating any key-value store with LMCache.

## Requirements

1. Implement a connector class inheriting from `BaseConnector` (see `base_connector.py`)
2. Place your module in Python import path

## Configuration

Specify module and class name by `remote_url` in `backend_type.yaml`, and the remote_url should contain
- **Module Path**: Specify the Python module path (e.g., `external_log_connector.lmc_external_log_connector`)
- **Connector Name**: Provide the class name of the connector (e.g., `ExternalLogConnector`)

## Example YAML Configuration

This example use lmc_external_log_connector as an example which is an internal lmcache remote connector. Reference [lmc_exernal_log_connector](https://github.com/opendataio/lmc_exernal_log_connector)

```yaml
remote_url: "external://host:0/external_log_connector.lmc_external_log_connector/?connector_name=ExternalLogConnector"
extra_config:
  ext_log_connector_support_ping: True
  ext_log_connector_health_interval: 10.0
  ext_log_connector_stuck_time: 6.0
```

## Start vLLM with the lmc_external_log_connector as an external connector

```shell
VLLM_USE_V1=0 \
LMCACHE_TRACK_USAGE=false \
LMCACHE_CONFIG_FILE=backend_type.yaml \
vllm serve /disc/f/models/opt-125m/ \
           --served-model-name "facebook/opt-125m" \
           --enforce-eager  \
           --port 8000 \
           --gpu-memory-utilization 0.8 \
           --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}' \
           --trust-remote-code
```
