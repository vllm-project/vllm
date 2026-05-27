# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cmd
import sys
import os
import base64
import json
import hashlib
import asyncio
import pathlib
import subprocess
import urllib.request
import urllib.error
import shlex
from argparse import ArgumentParser

from datetime import datetime, timezone

try:
    import uvloop
except ImportError:
    uvloop = None

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
# from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.logger import init_logger

logger = init_logger(__name__)

LICENSE_SERVER = {
    "host": [
        "https://auth.unieai.com",
        "https://uls.unieai.com",
        "https://13.114.141.202",
        "http://13.114.141.202",
    ],
    "info": """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA5Xtc4qwU2nxODyg3h2i8
2wMYofSUA9ZKpjaaLE1sbH8gJrij5KxSAShtLgq9I5O6FKtLfA+OVdYJnM7TzMS7
DIIgxabp3+x4NEbhUW2zi2Z4sX2eUFHlSDcy6xNoi6txk9KpHMKYt0QtHL7XGJPN
lULIvG5zwDTbJY3MpYiJW27U8qCbCGq/gnV3mva1NLNjL0vqTeiUQgPiwYakEPuJ
H0Yt5exYueMltRoTxRIOq2uK6KPJiQu0f9m1u/J3PXoTZN4WyySXealneN95wfeF
InxZBNLEBVHnJ1adWSAcmIdPLvljDixpMt57OPUa7dEDXO6e5mKF9aj9HcAER8BC
lQIDAQAB
-----END PUBLIC KEY-----"""
}

CACHE_DIR = pathlib.Path(os.environ.get("UNIEAI_CACHE_DIR", str(pathlib.Path.home() / ".unieai")))
CACHE_FILE = CACHE_DIR / "last_verified.json"

# ---------------------------------------------------------------------------
# Device fingerprint
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """Derive a deterministic session ID from NVIDIA GPU UUIDs.

    Runs ``nvidia-smi --query-gpu=uuid --format=csv,noheader``, collects all
    GPU UUIDs, sorts them alphabetically, concatenates them, and returns the
    SHA-256 hex digest as the session ID.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error("nvidia-smi failed (rc=%d): %s", result.returncode, result.stderr.strip())
            sys.exit(1)

        uuids = sorted(line.strip() for line in result.stdout.strip().splitlines() if line.strip())
        if not uuids:
            logger.error("nvidia-smi returned no GPU UUIDs.")
            sys.exit(1)

        combined = ",".join(uuids)
        session_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        logger.info("Session ID derived from %d GPU(s): %s", len(uuids), session_id)
        return session_id

    except FileNotFoundError:
        logger.error("nvidia-smi not found. NVIDIA drivers must be installed.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi timed out.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# License helpers
# ---------------------------------------------------------------------------

def _load_public_key():
    """Load the embedded RSA public key."""
    pem_data = LICENSE_SERVER["info"].encode("utf-8")
    return serialization.load_pem_public_key(pem_data)


def decrypt_license() -> dict:
    """Read UNIEAI_LICENSE env var, decrypt with public key, return license data.

    The env var holds a base64-encoded blob containing:
        {
            "data":      "<base64-encoded license JSON>",
            "signature": "<base64-encoded RSA signature of the data bytes>"
        }

    The public key is used to verify the signature (i.e. "decrypt"), proving
    the data was produced by the holder of the private key.  On success the
    inner license JSON is returned as a dict.  Expected fields:

        license_key  – e.g. "UNIE-TEST-001"
        session_id   – SHA-256 of sorted GPU UUIDs
        expires_at   – ISO-8601 date, e.g. "2026-12-31"
    """
    raw = os.environ.get("UNIEAI_LICENSE")
    if not raw:
        logger.error("UNIEAI_LICENSE environment variable is not set.")
        sys.exit(1)

    # ── Step 1: base64-decode the outer envelope ──────────────────────────
    try:
        envelope = json.loads(base64.b64decode(raw))
    except Exception as exc:
        logger.error("Failed to decode UNIEAI_LICENSE: %s", exc)
        sys.exit(1)

    data_b64: str | None = envelope.get("data")
    sig_b64: str | None = envelope.get("signature")

    if not data_b64 or not sig_b64:
        logger.error(
            "UNIEAI_LICENSE envelope must contain 'data' and 'signature' fields."
        )
        sys.exit(1)

    data_bytes = base64.b64decode(data_b64)
    sig_bytes = base64.b64decode(sig_b64)

    # ── Step 2: verify RSA signature (PSS + SHA-256) ─────────────────────
    try:
        public_key = _load_public_key()
        public_key.verify(
            sig_bytes,
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        logger.info("License decrypted and verified successfully.")
    except Exception as exc:
        logger.error("License verification failed (invalid signature): %s", exc)
        sys.exit(1)

    # ── Step 3: parse the inner JSON ─────────────────────────────────────
    try:
        license_data = json.loads(data_bytes)
    except Exception as exc:
        logger.error("Failed to parse license data JSON: %s", exc)
        sys.exit(1)

    required = ("license_key", "session_id", "expires_at")
    missing = [k for k in required if k not in license_data]
    if missing:
        logger.error("License data is missing required fields: %s", missing)
        sys.exit(1)

    logger.info(
        "License loaded — key=%s, session=%s, expires=%s",
        license_data["license_key"],
        license_data["session_id"],
        license_data["expires_at"],
    )
    return license_data

def check_online_server_availability() -> str: # return server host if reachable, else empty string
    """Check if any license server host is reachable."""
    for host in LICENSE_SERVER["host"]:
        url = f"{host}/api/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("License server is reachable at %s", host)
                    return host
        except Exception as exc:
            logger.debug("License server %s is unreachable: %s", host, exc)
            continue
    logger.warning("All license server hosts are unreachable.")
    return ""

# ---------------------------------------------------------------------------
# Online verification
# ---------------------------------------------------------------------------

def verify_license_online(license_key: str, session_id: str, host: str) -> bool:
    """Verify the license by POSTing to the server's heartbeat endpoint.

    Returns True if the server confirms ACTIVE + allowed, False if the server
    rejects the license, or raises an exception if all servers are unreachable.
    """
    if not host:
        logger.warning("No license server available for online verification.")
        return False

    payload = {
        "license_key": license_key,
        "machine_id": session_id,
    }
    # example payload in curl
    # curl -X POST http://13.114.141.202/api/licenses/heartbeat \
    #     -H "Content-Type: application/json" \
    #     -d '{"license_key":"UNIE-TEST-001","session_id":"test-machine-abc123"}'
    url = f"{host}/api/licenses/heartbeat"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))

        status = resp_body.get("status", "").upper()
        allowed = resp_body.get("allowed", False)

        if status == "ACTIVE" and allowed:
            logger.info(
                "License heartbeat OK via %s — status=%s, allowed=%s",
                host, status, allowed,
            )
            _write_verified_cache(license_key, resp_body)
            return True
        else:
            logger.warning("License rejected by %s — %s", host, resp_body)
            return False
    except Exception as exc:
        logger.warning("License verification failed for %s: %s", host, exc)
        return False

def _write_verified_cache(license_key: str, server_response: dict):
    """Persist the last successful verification to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = {
            "license_key": license_key,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "server_response": server_response,
        }
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        logger.info("Last-verified cache written to %s", CACHE_FILE)
    except Exception as exc:
        logger.warning("Could not write verification cache: %s", exc)


def _read_verified_cache() -> dict | None:
    """Read the last-verified cache, or None if unavailable."""
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read verification cache: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def check_license_validity(license_data: dict, online_ok: bool) -> bool:
    """Decide whether the license allows the server to launch.

    * If online verification succeeded → check ``expires_at`` is in the future.
    * If all servers were unreachable → allow launch only if the cached
      last-verified timestamp exists AND ``expires_at`` is still in the future.
    """
    now = datetime.now(timezone.utc)

    # Parse expiry
    try:
        expires_str = license_data["expires_at"]
        # Support both date-only ("2026-12-31") and full ISO datetime
        if "T" in expires_str:
            expires_at = datetime.fromisoformat(expires_str)
        else:
            expires_at = datetime.fromisoformat(expires_str + "T23:59:59+00:00")
        # Ensure timezone-aware
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.error("Invalid expires_at in license data: %s", exc)
        return False

    if expires_at <= now:
        logger.error(
            "License has expired (expires_at=%s, now=%s).",
            expires_at.isoformat(), now.isoformat(),
        )
        return False

    if online_ok:
        logger.info("License is valid (online verified, expires %s).", expires_at.date())
        return True

    # Offline fallback — check cache
    cache = _read_verified_cache()
    if cache is None:
        logger.error(
            "All license servers are unreachable and no previous verification "
            "cache exists. Cannot launch."
        )
        return False

    logger.info(
        "All license servers are unreachable, but license was last verified "
        "at %s and does not expire until %s. Allowing launch.",
        cache.get("verified_at", "unknown"),
        expires_at.date(),
    )
    return True


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    print("▗▖ ▗▖▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖\033[94m ▗▄▖ ▗▄▄▄▖\033[0m")
    print("▐▌ ▐▌▐▛▚▖▐▌  █  ▐▌   \033[94m▐▌ ▐▌  █\033[0m")
    print("▐▌ ▐▌▐▌ ▝▜▌  █  ▐▛▀▀▘\033[94m▐▛▀▜▌  █\033[0m")
    print("▝▚▄▞▘▐▌  ▐▌▗▄█▄▖▐▙▄▄▖\033[94m▐▌ ▐▌▗▄█▄▖\033[0m")
    print()
    print("▗▖ ▗▖▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖\033[91m▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▄▄▖  ▗▄▖\033[0m")
    print("▐▌ ▐▌▐▛▚▖▐▌  █  ▐▌   \033[91m  █  ▐▛▚▖▐▌▐▌   ▐▌ ▐▌▐▌ ▐▌\033[0m")
    print("▐▌ ▐▌▐▌ ▝▜▌  █  ▐▛▀▀▘\033[91m  █  ▐▌ ▝▜▌▐▛▀▀▘▐▛▀▚▖▐▛▀▜▌\033[0m")
    print("▝▚▄▞▘▐▌  ▐▌▗▄█▄▖▐▙▄▄▖\033[91m▗▄█▄▖▐▌  ▐▌▐▌   ▐▌ ▐▌▐▌ ▐▌\033[0m")
    parser = ArgumentParser(
        description="UnieInfra - UnieAI Licensed Inference Engine",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="UnieInfra Launch Command")
    serve.add_argument("model_name", help="The model tag to serve (optional if specified in config) (default: None)", default=None)

    unieconfig = subparsers.add_parser("unieconfig", help="Print license config and exit")
    unieconfig.add_argument("model_name", help="The model tag to serve (optional if specified in config) (default: None)", default=None)

    # License check before doing anything
    # mock license data
    # {
    #     "license_key": "UNIE-TEST-001",
    #     "session_id": "test-session-abc123", # using GPU UUID to get session_id
    #     "expires_at": "2026-12-31T23:59:59+00:00"
    # }
    license_data = decrypt_license()
    available_server = check_online_server_availability()
    online_ok = verify_license_online(license_data["license_key"], license_data["session_id"], available_server)

    if not online_ok:
        logger.info("Online license verification failed.")

    # Check offline license data is valid (e.g. not expired or GUP UUID mismatch) before allowing launch

    # Check expirey date
    # exmaple license_data["expires_at"] = "2026/06/18"
    expires_at_str = license_data["expires_at"]
    try:
        expires_at = datetime.fromisoformat(expires_at_str)
        if expires_at <= datetime.now(timezone.utc):
            logger.error("License has expired (expires_at=%s).", expires_at_str)
            sys.exit(1)
    except ValueError:
        logger.error("Invalid expires_at format in license data: %s", expires_at_str)
        sys.exit(1)

    if get_session_id() != license_data["session_id"]:
        logger.error(
            "GPU UUID mismatch: license session_id=%s, but current machine session_id=%s.",
            license_data["session_id"], get_session_id(),
        )
        sys.exit(1)

    args, unknown_args = parser.parse_known_args()

    if args.command == "serve":
        if "--easy" in unknown_args:
            # remove --easy
            unknown_args = [arg for arg in unknown_args if arg != "--easy"]
        else:
            is_overlap = any(
                arg == "--kv-cache-dtype" or arg.startswith("--kv-cache-dtype=")
                for arg in unknown_args
            )
            if not is_overlap:
                unknown_args += ["--kv-cache-dtype", "fp8"]
        cmd = ["vllm", "serve", args.model_name] + unknown_args
        # logger.info("Running UnieConfig with command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, shell=False, text=True)
    if args.command == "unieconfig":
        unieconfig_args = [
            "--score-concurrencies", "1,8,64,256", "--n-trials", "20", "-o", "benchmarks/results"
        ]
        unknown_cmd = " ".join(unknown_args)
        serve_cmd = f"vllm serve {args.model_name} {unknown_cmd}"
        cmd = [
            "vllm",
            "serve-optuna",
            "--serve-cmd",
            shlex.quote(serve_cmd),
        ] + unieconfig_args
        # logger.info("Running UnieConfig with command: %s", " ".join(cmd))
        subprocess.run(" ".join(cmd), check=True, shell=True, text=True)

if __name__ == "__main__":
    main()
