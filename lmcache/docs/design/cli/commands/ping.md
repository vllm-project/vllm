# `lmcache ping` — Design & Implementation Plan

**Status:** Proposal  |  **Date:** 2026-03-23

## Context

The CLI framework (Phase 0) and `describe kvcache` (Phase 1, PR #2825) are
complete. The next Phase 1 command is `lmcache ping`, a pure liveness check for
the LMCache server process and the vLLM server process.

---

## Command UX

```bash
$ lmcache ping kvcache --url http://localhost:8080

======= Ping KV Cache =======
Status:                  OK
Round trip time (ms):    0.42
==============================
```

```bash
$ lmcache ping engine --url http://localhost:8000

======== Ping Engine =========
Status:                  OK
Round trip time (ms):    12.3
==============================
```

JSON output:

```bash
$ lmcache ping kvcache --url http://localhost:8080 --format json
```

```json
{
  "title": "Ping KV Cache",
  "metrics": {
    "status": "OK",
    "round_trip_time_ms": 0.42
  }
}
```

```bash
$ lmcache ping engine --url http://localhost:8000 --format json
```

```json
{
  "title": "Ping Engine",
  "metrics": {
    "status": "OK",
    "round_trip_time_ms": 12.3
  }
}
```

---

## Design Decisions

### 1. Sub-target as positional argument

```
lmcache ping kvcache --url http://localhost:8080
lmcache ping engine  --url http://localhost:8000
```

Uses a positional `target` argument with `choices=["kvcache", "engine"]`.
Matches the `ping {kvcache,engine}` pattern in [commands.md](../commands.md).

### 2. Both targets use HTTP

Both `ping kvcache` and `ping engine` use a simple HTTP GET to the respective
server's health endpoint. No ZMQ client is needed.

| Target | Server process | Endpoint | Healthy | Unhealthy |
|--------|---------------|----------|---------|-----------|
| `kvcache` | LMCache MP server | `GET /healthcheck` | 200 `{"status": "healthy"}` | 503 `{"status": "unhealthy", "reason": "..."}` |
| `engine` | vLLM server | `GET /health` | 200 (empty body) | 503 (empty body) |

### 3. `--url` defaults

| Target | Default URL |
|--------|-------------|
| `kvcache` | `http://localhost:8080` |
| `engine` | `http://localhost:8000` |

The default for `kvcache` matches `describe kvcache`. The default for `engine`
matches the standard vLLM serving port.

### 4. Round-trip time measurement

The round-trip time measures only the HTTP request-response cycle, using
`time.monotonic()` around the `urllib.request.urlopen()` call. This excludes
Python startup, argument parsing, and output formatting overhead.

### 5. HTTP client: stdlib `urllib`

Same as `describe` — uses `urllib.request` with no new dependencies.

### 6. Error handling

| Condition | Status | Exit code |
|-----------|--------|-----------|
| 200 response | OK | 0 |
| 503 response | FAIL | 1 |
| Connection refused / timeout | FAIL (with error detail) | 1 |
| Other HTTP error | FAIL (with HTTP status) | 1 |

On failure, `status` is reported as `"FAIL"` and the round-trip time is still
reported (it shows how long we waited before the error). A detail message is
printed to stderr.

---

## CLI Implementation

### New file: `lmcache/cli/commands/ping.py`

```python
class PingCommand(BaseCommand):
    name() → "ping"
    help() → "Ping LMCache or vLLM server (liveness check)."

    add_arguments(parser):
        parser.add_argument("target", choices=["kvcache", "engine"],
                            help="What to ping.")
        parser.add_argument("--url", default=None,
                            help="Server URL (default: http://localhost:8080 "
                                 "for kvcache, http://localhost:8000 for engine)")

    execute(args):
        url = args.url or DEFAULT_URLS[args.target]
        url = normalize_url(url)
        endpoint = HEALTH_ENDPOINTS[args.target]  # "/healthcheck" or "/health"
        title = TITLES[args.target]               # "Ping KV Cache" or "Ping Engine"

        status, rtt_ms, error = ping(f"{url}{endpoint}")

        metrics = self.create_metrics(title, args, width=30)
        metrics.add("status", "Status", status)
        metrics.add("round_trip_time_ms", "Round trip time (ms)", round(rtt_ms, 2))
        metrics.emit()

        if error:
            print(error, file=sys.stderr)
            sys.exit(1)
```

Module-level helper:

```python
HEALTH_ENDPOINTS = {"kvcache": "/healthcheck", "engine": "/health"}
DEFAULT_URLS = {"kvcache": "http://localhost:8080", "engine": "http://localhost:8000"}
TITLES = {"kvcache": "Ping KV Cache", "engine": "Ping Engine"}

def ping(url: str, timeout: int = 10) -> tuple[str, float, str | None]:
    """GET *url* and return (status, rtt_ms, error_msg).

    Returns:
        ("OK", rtt_ms, None) on 200.
        ("FAIL", rtt_ms, detail) on error.
    """
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            rtt_ms = (time.monotonic() - start) * 1000
            if resp.status == 200:
                return ("OK", rtt_ms, None)
            return ("FAIL", rtt_ms, f"HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        rtt_ms = (time.monotonic() - start) * 1000
        return ("FAIL", rtt_ms, f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        rtt_ms = (time.monotonic() - start) * 1000
        reason = getattr(exc, "reason", str(exc))
        return ("FAIL", rtt_ms, f"Cannot connect to {url}: {reason}")
```

Reuses `normalize_url()` from `describe.py` (import it, or move to a shared
`lmcache/cli/utils.py` if preferred).

### Modify: `lmcache/cli/commands/__init__.py`

```python
from lmcache.cli.commands.ping import PingCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
    DescribeCommand(),
    PingCommand(),
]
```

---

## Verification

1. **Unit tests** (`tests/cli/test_ping.py`):
   - Test `ping()` helper with a real local `HTTPServer` returning 200 and 503.
   - Test connection refused (unreachable port) returns `"FAIL"`.
   - Test `PingCommand.execute()` end-to-end with mocked `ping()` for both
     targets, verifying JSON output fields.
   - Test `--url` default resolution per target.
2. **Manual tests:**
   ```bash
   lmcache ping kvcache --url http://localhost:8080
   lmcache ping kvcache --url http://localhost:8080 --format json
   lmcache ping engine  --url http://localhost:8000
   lmcache ping engine  --url http://localhost:8000 --format json
   lmcache ping kvcache --url http://localhost:9999   # connection refused → FAIL, exit 1
   lmcache ping engine  --url http://localhost:9999   # connection refused → FAIL, exit 1
   lmcache ping kvcache                               # uses default http://localhost:8080
   lmcache ping engine                                # uses default http://localhost:8000
   ```
