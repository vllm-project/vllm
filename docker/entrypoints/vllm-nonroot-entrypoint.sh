#!/bin/sh
# Entrypoint wrapper for the opt-in `vllm-openai-nonroot` image.
#
# The image also ships a `vllm` user (UID 2000, GID 0) with HOME /home/vllm
# and a group-0-writable home directory. When the container is launched with
# `--user 2000:0` (or any other UID in group 0) the passwd entry is enough on
# its own: Docker picks up HOME=/home/vllm, getpass.getuser() resolves to
# "vllm", and every cache dir (HF, Triton, Inductor, vLLM, Numba, Outlines)
# that defaults to `$HOME/.cache/...` lands in a writable location.
#
# This wrapper exists for the *arbitrary-UID* case (e.g. OpenShift's
# `runAsUser: 1000540000` Restricted Pod Security Standard) where the caller
# UID is not in /etc/passwd at all. In that case:
#   * $HOME may be unset or resolve to "/" (unwritable).
#   * getpass.getuser() falls back to pwd.getpwuid() -> KeyError.
#
# The wrapper re-points $HOME to /home/vllm when writable, /tmp/vllm-home
# otherwise, and defaults $USER to "vllm" so the pwd-lookup path is never
# taken. Everything else is forwarded to `vllm serve`.
#
# Non-empty caller-set env vars (HOME, USER, LOGNAME) are preserved, so
# existing K8s manifests and `docker run -e ...` keep working unchanged.
# Unset or empty values fall through to the wrapper's defaults, matching
# what shell code typically expects from "unset".

set -eu

if [ -z "${HOME:-}" ] || [ ! -w "${HOME}" ]; then
    if [ -w /home/vllm ]; then
        export HOME=/home/vllm
    else
        export HOME=/tmp/vllm-home
        mkdir -p "$HOME" 2>/dev/null || true
    fi
fi

# If the current working directory is not writable (e.g. an arbitrary UID in
# a non-0 GID landing on the image's WORKDIR=/home/vllm, or a misconfigured
# `-w /readonly/path`), chdir into $HOME so vllm's cwd-relative path probe
# (e.g. "is this model arg a local path?") doesn't crash with a confusing
# PermissionError.
#
# We *only* chdir when the CWD is actually unusable. This preserves
# caller-provided CWDs for the common case of `docker run -w /models ...`
# plus relative argv like `--model ./llama.gguf`, `--chat-template
# ./t.jinja`, relative TLS cert paths, etc.
if [ ! -w . ]; then
    cd "$HOME"
fi

# getpass.getuser() prefers $USER/$LOGNAME/etc. before hitting getpwuid();
# setting it here makes the "UID not in passwd" path a no-op for everything
# in the process tree.
if [ -z "${USER:-}" ]; then
    export USER=vllm
fi
if [ -z "${LOGNAME:-}" ]; then
    export LOGNAME="$USER"
fi

# Shell-level tooling (`whoami`, bash's `\u` prompt, `id -un`, `sudo`) does
# NOT consult $USER; it calls getpwuid(geteuid()) directly. For arbitrary
# runtime UIDs in OpenShift-style deploys this returns "I have no name!".
# If /etc/passwd is group-0 writable (set at build time) and doesn't yet
# have an entry for this UID, append a synthetic one so every downstream
# consumer sees a consistent "vllm" identity.
#
# We parse the passwd file directly instead of calling `getent` because
# the container's NSS is typically just files anyway, and this lets us
# unit-test via the VLLM_PASSWD_FILE hook (undocumented; production uses
# /etc/passwd).
_passwd_file="${VLLM_PASSWD_FILE:-/etc/passwd}"
_uid="$(id -u)"
if [ -w "$_passwd_file" ] \
    && ! awk -F: -v u="$_uid" '$3==u {found=1; exit} END {exit !found}' "$_passwd_file" 2>/dev/null; then
    printf 'vllm:x:%s:%s:vllm:%s:/bin/bash\n' \
        "$_uid" "$(id -g)" "$HOME" >> "$_passwd_file"
fi
unset _uid _passwd_file

exec vllm serve "$@"
