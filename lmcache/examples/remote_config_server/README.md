# Remote Config Server Example

This example provides a reference implementation of a remote config server
that can be used with LMCache's dynamic configuration feature.

## Overview

LMCache supports fetching configuration from a remote config service at startup.
This allows centralized management of LMCache configurations across multiple workers.

## Configuration Fields

To enable remote configuration, add these fields to your LMCache config file:

```yaml
# URL of the remote config service
remote_config_url: "http://localhost:8088/config"

# Optional: Application ID for identifying different applications
app_id: "my-app-001"
```

## Protocol Specification

### Request

The LMCache worker sends a **POST** request to the `remote_config_url`:

- **Method**: POST
- **Query Parameters**: `?appId=<app_id>` (if `app_id` is configured)
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
    "current_config": {
        "chunk_size": 256,
        "local_device": "cpu",
        ...
    },
    "env_variables": {
        "LMCACHE_CONFIG_FILE": "/path/to/config.yaml",
        "HOME": "/home/user",
        ...
    }
}
```

### Response

The config server should return a JSON response:

```json
{
  "configs": [
    {
      "key": "chunk_size",
      "override": false,
      "value": 1024
    },
    {
      "key": "max_local_cpu_size",
      "override": false,
      "value": 2
    },
    {
      "key": "lmcache_worker_heartbeat_time",
      "override": true,
      "value": "14"
    },
    {
      "key": "extra_config",
      "override": true,
      "value": "{\"internal_api_server_access_log\": true, \"internal_api_server_log_level\":\"info\", \"save_only_first_rank\": true, \"first_rank_max_local_cpu_size\": 1.1}"
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `configs` | array | List of configuration items to apply |
| `configs[].key` | string | Configuration key name (must match LMCache config fields) |
| `configs[].value` | any | Value to set for this configuration |
| `configs[].override` | boolean | If `true`, always override. If `false`, only apply when current value is `None` |

## Running the Example

1. Install dependencies:
```bash
pip install flask
```

2. Start the config server:
```bash
python config_server.py
```

3. Configure LMCache to use this server:
```yaml
# example.yaml
chunk_size: 256
local_device: "cpu"
remote_config_url: "http://localhost:8088/config"
app_id: "test-app"
```

4. Start your LMCache-enabled application with the config file.

## Customization

You can customize the `config_server.py` to:
- Fetch configurations from a database
- Apply different configs based on `app_id`
- Implement access control based on environment variables
- Add logging and monitoring

5. How to test

Start standalone LMCache server to check if the config server and remote config are working properly.

See the document of `standalone_starter` for more details.
