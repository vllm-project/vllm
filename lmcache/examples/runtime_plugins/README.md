# LMCache Runtime Plugin System Documentation

## Overview

The LMCache runtime plugin system allows users to extend functionality by running custom scripts or programs during cache operations. Plugins can be written in any language (Python, Bash, etc.) and are managed by the `RuntimePluginLauncher` class.

Based on this, we hope users in our community can contribute more runtime plugins, e.g.
- Start metric reporter to centralized metric system.
- Start log reporter to centralized log collect and query system.
- Report customized process level metrics to the alert system.
- Heartbeat to a health monitor system or a service discover system.
- ...

## Configuration
Runtime plugins are configured through the following methods:

1. **Environment Variables**:
   - `LMCACHE_RUNTIME_PLUGIN_ROLE`: The role of the current process (e.g., `SCHEDULER`, `WORKER`)
   - `LMCACHE_RUNTIME_PLUGIN_CONFIG`: JSON string containing the plugin configuration
   - `LMCACHE_RUNTIME_PLUGIN_WORKER_ID`: The worker id of current process
   - `LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT`: The total worker count of this cluster

2. **Configuration File**:
   Runtime plugins can be specified in the `lmcache.yaml` file under the `runtime_plugin_locations` field:
   ```yaml
   runtime_plugin_locations: ["/path/to/plugins"]
   ```
3. **Pass more parameters via lmcache extra_config**
   You can Pass more parameters via specify extra_config within `lmcache.yaml`.

## Plugin Naming Rules
Runtime plugin filenames determine which roles/worker_id they run on:

**Role-Specific Plugins**:
   - Format: `<ROLE>[_<WORKER_ID>][_<DESCRIPTION>].<EXTENSION>`
   - Examples:
     - `scheduler_foo_plugin.py`: Runs only on `SCHEDULER` role
     - `worker_0_test.sh`: Runs only on `WORKER` with `worker_id=0`
     - `all_plugin.sh`: Runs on all nodes

Notes:
- Role names are case-insensitive (e.g., `worker` = `WORKER`)
- Worker ID must be a numeric value if specified

## Plugin Execution
Runtime plugins are executed as follows:

1. **Interpreter Detection**:
   - The first line (shebang) determines the interpreter:
     ```python
     #!/opt/venv/bin/python
     ```
   - Fallback interpreters:
     - `.py` files: `python`
     - `.sh` files: `bash`

2. **Output Capture**:
   - Runtime plugin stdout/stderr is captured continuously
   - Output is logged with the plugin name as prefix

3. **Process Management**:
   - Runtime plugins are launched as subprocesses
   - All runtime plugins are terminated when the parent process exits

## Example Plugins
1. Python Plugin (`scheduler_foo_plugin.py`)
2. Bash Plugin (`all_plugin.sh`)

## Best Practices
1. Keep runtime plugins lightweight
2. Use descriptive names
3. Handle errors gracefully
4. Include shebang for portability
