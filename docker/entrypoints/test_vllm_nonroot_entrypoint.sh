#!/bin/sh
# Shell-level unit test for vllm-nonroot-entrypoint.sh.
#
# Runs on the host (no Docker, no GPU) by stubbing `vllm` with a shim that
# dumps its env + argv instead of actually serving. Exercises the wrapper's
# HOME/USER fallback behavior that can't be easily tested from buildkite
# (which would need a GPU to run `vllm serve --help`).
#
# Usage:
#   bash docker/entrypoints/test_vllm_nonroot_entrypoint.sh
# Exits non-zero on the first failed assertion.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WRAPPER="${SCRIPT_DIR}/vllm-nonroot-entrypoint.sh"

if [ ! -x "$WRAPPER" ]; then
    echo "FAIL: wrapper not found or not executable: $WRAPPER" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Stub `vllm` on PATH. It dumps env + argv + cwd to stdout so we can assert.
mkdir -p "$WORKDIR/bin"
cat > "$WORKDIR/bin/vllm" <<'EOF'
#!/bin/sh
echo "ARGV=$*"
echo "HOME=${HOME-__unset__}"
echo "USER=${USER-__unset__}"
echo "LOGNAME=${LOGNAME-__unset__}"
echo "PWD=$(pwd)"
EOF
chmod +x "$WORKDIR/bin/vllm"

run_wrapper() {
    # Usage: run_wrapper <output_file> <env_kv>... -- <wrapper_arg>...
    _out="$1"; shift
    _env=""
    while [ "${1:-}" != "--" ]; do
        _env="$_env $1"; shift
    done
    shift
    env -i PATH="$WORKDIR/bin:/usr/bin:/bin" $_env "$WRAPPER" "$@" > "$_out"
}

fail() { echo "FAIL: $*" >&2; echo "--- stdout ---" >&2; cat "$1" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Case 1: writable HOME and USER both set -> wrapper must leave them alone.
# -----------------------------------------------------------------------------
case1_home="$WORKDIR/case1-home"
mkdir -p "$case1_home"
out="$WORKDIR/case1.out"
run_wrapper "$out" "HOME=$case1_home" "USER=alice" "LOGNAME=alice" -- --model foo
grep -q "^HOME=$case1_home\$" "$out" || fail "$out" "case1: HOME not preserved"
grep -q "^USER=alice\$" "$out" || fail "$out" "case1: USER not preserved"
grep -q "^LOGNAME=alice\$" "$out" || fail "$out" "case1: LOGNAME not preserved"
grep -q "^ARGV=serve --model foo\$" "$out" || fail "$out" "case1: ARGV wrong"
echo "PASS: case1 (writable HOME + USER preserved)"

# -----------------------------------------------------------------------------
# Case 2: HOME unset -> falls back to /home/vllm if writable, else /tmp/vllm-home.
# -----------------------------------------------------------------------------
fake_vllm_home="$WORKDIR/fake-home-vllm"
mkdir -p "$fake_vllm_home"
# Temporarily shadow /home/vllm in the wrapper by setting the working dir so
# "[ -w /home/vllm ]" resolves based on whether a real /home/vllm exists and
# is writable. On dev machines /home/vllm typically does NOT exist, so the
# wrapper should fall to /tmp/vllm-home.
out="$WORKDIR/case2.out"
run_wrapper "$out" -- --model bar
if [ -w /home/vllm ]; then
    expected_home="/home/vllm"
else
    expected_home="/tmp/vllm-home"
fi
grep -q "^HOME=$expected_home\$" "$out" || fail "$out" "case2: HOME not set to $expected_home"
grep -q "^USER=vllm\$" "$out" || fail "$out" "case2: USER not defaulted to vllm"
grep -q "^LOGNAME=vllm\$" "$out" || fail "$out" "case2: LOGNAME not defaulted to vllm"
grep -q "^ARGV=serve --model bar\$" "$out" || fail "$out" "case2: ARGV wrong"
echo "PASS: case2 (unset HOME falls back to $expected_home, USER defaulted)"

# -----------------------------------------------------------------------------
# Case 3: HOME set but unwritable -> must also fall back.
# -----------------------------------------------------------------------------
ro_home="$WORKDIR/ro-home"
mkdir -p "$ro_home"
chmod 0500 "$ro_home"
out="$WORKDIR/case3.out"
run_wrapper "$out" "HOME=$ro_home" -- --model baz
grep -q "^HOME=$expected_home\$" "$out" || fail "$out" "case3: HOME not overridden from unwritable"
grep -q "^USER=vllm\$" "$out" || fail "$out" "case3: USER not defaulted"
chmod 0700 "$ro_home"
echo "PASS: case3 (unwritable HOME overridden)"

# -----------------------------------------------------------------------------
# Case 4: USER set but LOGNAME unset -> LOGNAME mirrors USER.
# -----------------------------------------------------------------------------
case4_home="$WORKDIR/case4-home"
mkdir -p "$case4_home"
out="$WORKDIR/case4.out"
run_wrapper "$out" "HOME=$case4_home" "USER=carol" -- --model qux
grep -q "^USER=carol\$" "$out" || fail "$out" "case4: USER not preserved"
grep -q "^LOGNAME=carol\$" "$out" || fail "$out" "case4: LOGNAME not mirrored from USER"
echo "PASS: case4 (LOGNAME mirrors USER when unset)"

# -----------------------------------------------------------------------------
# Case 5: /etc/passwd is writable AND the current UID is not in it -> wrapper
# appends a synthetic entry. Uses the VLLM_PASSWD_FILE test hook so we don't
# touch the real /etc/passwd.
# -----------------------------------------------------------------------------
fake_passwd="$WORKDIR/fake-passwd"
: > "$fake_passwd"  # empty file, current UID definitely not present
case5_home="$WORKDIR/case5-home"
mkdir -p "$case5_home"
out="$WORKDIR/case5.out"
run_wrapper "$out" "HOME=$case5_home" "VLLM_PASSWD_FILE=$fake_passwd" -- --model foo
current_uid="$(id -u)"
current_gid="$(id -g)"
expected_line="vllm:x:${current_uid}:${current_gid}:vllm:${case5_home}:/bin/bash"
grep -Fx "$expected_line" "$fake_passwd" > /dev/null \
    || { echo "FAIL: case5: expected line not found in fake passwd:"; echo "  expected: $expected_line"; echo "  file contents:"; cat "$fake_passwd"; exit 1; }
echo "PASS: case5 (passwd entry appended for arbitrary UID)"

# -----------------------------------------------------------------------------
# Case 6: /etc/passwd is writable but current UID already has an entry ->
# wrapper must NOT duplicate the entry.
# -----------------------------------------------------------------------------
fake_passwd="$WORKDIR/fake-passwd-prepopulated"
printf 'vllm:x:%s:%s:vllm:/home/vllm:/bin/bash\n' "$current_uid" "$current_gid" > "$fake_passwd"
out="$WORKDIR/case6.out"
run_wrapper "$out" "HOME=$case5_home" "VLLM_PASSWD_FILE=$fake_passwd" -- --model foo
line_count="$(wc -l < "$fake_passwd")"
# NOTE: wc may count 0 or 1 depending on trailing newline; accept 1.
# More robust: count lines matching our UID.
uid_lines="$(grep -c ":${current_uid}:" "$fake_passwd" || true)"
[ "$uid_lines" = "1" ] \
    || { echo "FAIL: case6: expected exactly one entry for UID $current_uid, got $uid_lines"; cat "$fake_passwd"; exit 1; }
echo "PASS: case6 (existing passwd entry not duplicated)"

# -----------------------------------------------------------------------------
# Case 7: /etc/passwd is NOT writable -> wrapper must NOT crash, just skip.
# Skipped when running as root, because root's DAC override means [ -w ... ]
# is always true regardless of mode bits -- the case can't be simulated.
# In the real deployment (non-root UID inside the container) this IS the
# relevant behavior and is what `_passwd_file is not writable` encodes.
# -----------------------------------------------------------------------------
if [ "$(id -u)" = "0" ]; then
    echo "SKIP: case7 (running as root; DAC override makes unwritable check meaningless)"
else
    fake_passwd="$WORKDIR/ro-passwd"
    : > "$fake_passwd"
    chmod 0444 "$fake_passwd"
    out="$WORKDIR/case7.out"
    run_wrapper "$out" "HOME=$case5_home" "VLLM_PASSWD_FILE=$fake_passwd" -- --model foo
    # File must remain empty (no write happened) and the wrapper exec'd
    # `vllm serve` successfully (stdout contains ARGV line).
    [ ! -s "$fake_passwd" ] \
        || { echo "FAIL: case7: RO passwd file was modified"; cat "$fake_passwd"; exit 1; }
    grep -q "^ARGV=serve --model foo\$" "$out" || fail "$out" "case7: wrapper didn't exec vllm"
    chmod 0600 "$fake_passwd"
    echo "PASS: case7 (unwritable passwd file tolerated)"
fi

# -----------------------------------------------------------------------------
# Case 8: caller's writable CWD is preserved — wrapper must NOT chdir to HOME
# when cwd is usable. Protects relative-path workflows like
# `docker run -w /models ... --model ./llama.gguf`.
# -----------------------------------------------------------------------------
case8_home="$WORKDIR/case8-home"
mkdir -p "$case8_home"
case8_cwd="$WORKDIR/case8-cwd"
mkdir -p "$case8_cwd"
out="$WORKDIR/case8.out"
(cd "$case8_cwd" && run_wrapper "$out" "HOME=$case8_home" "USER=alice" "LOGNAME=alice" -- --model ./relpath)
grep -q "^PWD=$case8_cwd\$" "$out" \
    || fail "$out" "case8: writable cwd not preserved (got $(grep '^PWD=' "$out"))"
grep -q "^ARGV=serve --model \\./relpath\$" "$out" \
    || fail "$out" "case8: relative argv not preserved"
echo "PASS: case8 (writable cwd preserved; relative argv still resolves from caller's cwd)"

# -----------------------------------------------------------------------------
# Case 9: unwritable CWD is overridden — wrapper falls back to HOME so vllm's
# cwd-relative path probe doesn't crash under arbitrary UIDs landing on a
# readonly WORKDIR. Skipped as root (DAC override).
# -----------------------------------------------------------------------------
if [ "$(id -u)" = "0" ]; then
    echo "SKIP: case9 (running as root; DAC override makes unwritable cwd meaningless)"
else
    case9_home="$WORKDIR/case9-home"
    mkdir -p "$case9_home"
    case9_ro="$WORKDIR/case9-ro"
    mkdir -p "$case9_ro"
    chmod 0555 "$case9_ro"
    out="$WORKDIR/case9.out"
    (cd "$case9_ro" && run_wrapper "$out" "HOME=$case9_home" "USER=alice" "LOGNAME=alice" -- --model foo)
    grep -q "^PWD=$case9_home\$" "$out" \
        || fail "$out" "case9: unwritable cwd not overridden to HOME (got $(grep '^PWD=' "$out"))"
    chmod 0700 "$case9_ro"
    echo "PASS: case9 (unwritable cwd falls back to \$HOME)"
fi

echo ""
echo "ALL CASES PASSED."
