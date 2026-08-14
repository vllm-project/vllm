#!/bin/bash

# Process and container teardown helpers for the AMD CI runner. This file is
# sourced by run-amd-test.sh so the runner can keep diagnostics and teardown in
# one EXIT path while testing these primitives independently.

VLLM_CI_ACTIVE_PID=""
VLLM_CI_ACTIVE_PGID=""
VLLM_CI_LAUNCHING_CHILD=0
VLLM_CI_PENDING_SIGNAL_STATUS=0

vllm_ci_parse_bounded_integer() {
  local name=$1
  local value=$2
  local minimum=$3
  local maximum=$4
  local -n destination=$5
  local normalized=0

  # Reject leading zeroes and bound the digit count before arithmetic. Bash
  # otherwise treats values such as 08 as octal and recursively evaluates
  # arithmetic input taken from the environment.
  if [[ ! "${value}" =~ ^(0|[1-9][0-9]{0,8})$ ]]; then
    echo "${name} must be a base-10 integer from ${minimum} to ${maximum}; got ${value}" >&2
    return 1
  fi
  normalized=$((10#${value}))
  if ((normalized < minimum || normalized > maximum)); then
    echo "${name} must be a base-10 integer from ${minimum} to ${maximum}; got ${value}" >&2
    return 1
  fi
  destination=${normalized}
}

vllm_ci_require_process_tools() {
  command -v setsid >/dev/null 2>&1 || {
    echo "setsid is required for CI teardown" >&2
    return 1
  }
  [[ -r "/proc/$$/stat" ]] || {
    echo "/proc process metadata is required for CI teardown" >&2
    return 1
  }
}

vllm_ci_sanitize_resource_id() {
  local value=${1//[^A-Za-z0-9_.-]/-}

  value="${value:0:80}"
  printf '%s\n' "${value:-local}"
}

vllm_ci_monotonic_deciseconds() {
  local -n destination=$1
  local seconds=""
  local fraction=""
  local _ignored=""

  if IFS='. ' read -r seconds fraction _ignored < /proc/uptime; then
    destination=$((10#${seconds} * 10 + 10#${fraction:0:1}))
  else
    destination=$((SECONDS * 10))
  fi
}

vllm_ci_read_process() {
  local pid=$1
  local -n destination_state=$2
  local -n destination_group=$3
  local fields=""
  local stat_line=""

  destination_state=""
  destination_group=""
  IFS= read -r stat_line 2>/dev/null < "/proc/${pid}/stat" || return 1
  fields="${stat_line##*) }"
  destination_state="${fields%% *}"
  fields="${fields#* }"
  fields="${fields#* }"
  destination_group="${fields%% *}"
  [[ -n "${destination_state}" && "${destination_group}" =~ ^[0-9]+$ ]]
}

vllm_ci_process_is_live() {
  local state=""
  local process_group=""

  vllm_ci_read_process "$1" state process_group || return 1
  [[ "${state}" != "Z" ]]
}

vllm_ci_process_group_is_live() {
  local pgid=$1
  local path=""
  local pid=""
  local state=""
  local process_group=""

  kill -0 -- "-${pgid}" 2>/dev/null || return 1
  for path in /proc/[0-9]*/stat; do
    pid="${path#/proc/}"
    pid="${pid%/stat}"
    vllm_ci_read_process "${pid}" state process_group || continue
    if [[ "${process_group}" == "${pgid}" && "${state}" != "Z" ]]; then
      return 0
    fi
  done
  return 1
}

vllm_ci_terminate_process_group() {
  local pgid=$1
  local grace_seconds=$2
  local deadline=0
  local now=0
  local own_group=""
  local _own_state=""

  [[ -z "${pgid}" ]] && return 0
  if [[ ! "${pgid}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Refusing to signal invalid process group: ${pgid}" >&2
    return 1
  fi
  vllm_ci_read_process "$$" _own_state own_group || own_group=""
  if [[ "${pgid}" == "${own_group}" ]]; then
    echo "Refusing to signal the CI wrapper's process group" >&2
    return 1
  fi
  vllm_ci_process_group_is_live "${pgid}" || return 0

  echo "Gracefully terminating process group ${pgid}" >&2
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  vllm_ci_monotonic_deciseconds now
  deadline=$((now + grace_seconds * 10))
  while :; do
    vllm_ci_process_group_is_live "${pgid}" || return 0
    vllm_ci_monotonic_deciseconds now
    ((now >= deadline)) && break
    sleep 0.1
  done

  echo "Forcefully terminating process group ${pgid}" >&2
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  vllm_ci_monotonic_deciseconds now
  deadline=$((now + 10))
  while :; do
    vllm_ci_process_group_is_live "${pgid}" || return 0
    vllm_ci_monotonic_deciseconds now
    ((now >= deadline)) && break
    sleep 0.1
  done
  echo "Process group ${pgid} survived SIGKILL" >&2
  return 1
}

vllm_ci_find_tagged_pids() {
  local token=$1
  local entry=""
  local path=""
  local pid=""

  for path in /proc/[0-9]*/environ; do
    [[ -r "${path}" ]] || continue
    pid="${path#/proc/}"
    pid="${pid%/environ}"
    [[ "${pid}" == "$$" ]] && continue
    {
      while IFS= read -r -d '' entry; do
        if [[ "${entry}" == "VLLM_CI_PROCESS_TOKEN=${token}" ]]; then
          printf '%s\n' "${pid}"
          break
        fi
      done < "${path}"
    } 2>/dev/null
  done
}

vllm_ci_terminate_tagged_processes() {
  local token=$1
  local grace_seconds=$2
  local deadline=0
  local now=0
  local -a pids=()

  [[ -z "${token}" ]] && return 0
  vllm_ci_monotonic_deciseconds now
  deadline=$((now + grace_seconds * 10))
  while :; do
    mapfile -t pids < <(vllm_ci_find_tagged_pids "${token}")
    ((${#pids[@]})) || return 0
    kill -TERM "${pids[@]}" 2>/dev/null || true
    vllm_ci_monotonic_deciseconds now
    ((now >= deadline)) && break
    sleep 0.1
  done

  echo "Forcefully terminating tagged CI processes" >&2
  vllm_ci_monotonic_deciseconds now
  deadline=$((now + 10))
  while :; do
    mapfile -t pids < <(vllm_ci_find_tagged_pids "${token}")
    ((${#pids[@]})) || return 0
    kill -KILL "${pids[@]}" 2>/dev/null || true
    vllm_ci_monotonic_deciseconds now
    ((now >= deadline)) && break
    sleep 0.1
  done
  echo "Tagged CI processes survived SIGKILL: ${pids[*]}" >&2
  return 1
}

vllm_ci_request_exit() {
  local status=$1

  if ((VLLM_CI_LAUNCHING_CHILD)); then
    VLLM_CI_PENDING_SIGNAL_STATUS="${status}"
    return
  fi
  exit "${status}"
}

vllm_ci_run_tracked_command() {
  local timeout_seconds=$1
  local grace_seconds=$2
  local actual_group=""
  local deadline=0
  local group_ready=0
  local _process_state=""
  local now=0
  local status=0
  local timed_out=0
  shift 2

  VLLM_CI_LAUNCHING_CHILD=1
  setsid "$@" <&0 &
  VLLM_CI_ACTIVE_PID=$!
  VLLM_CI_ACTIVE_PGID="${VLLM_CI_ACTIVE_PID}"

  vllm_ci_monotonic_deciseconds now
  deadline=$((now + 10))
  while vllm_ci_read_process \
    "${VLLM_CI_ACTIVE_PID}" _process_state actual_group; do
    if [[ "${actual_group}" == "${VLLM_CI_ACTIVE_PID}" ]]; then
      group_ready=1
      break
    fi
    vllm_ci_monotonic_deciseconds now
    ((now >= deadline)) && break
    sleep 0.01
  done
  VLLM_CI_LAUNCHING_CHILD=0

  if vllm_ci_process_is_live "${VLLM_CI_ACTIVE_PID}" \
    && ((group_ready == 0)); then
    echo "Command did not establish process group" >&2
    kill -KILL "${VLLM_CI_ACTIVE_PID}" 2>/dev/null || true
    status=1
  fi
  if ((VLLM_CI_PENDING_SIGNAL_STATUS)); then
    status="${VLLM_CI_PENDING_SIGNAL_STATUS}"
    VLLM_CI_PENDING_SIGNAL_STATUS=0
    return "${status}"
  fi
  ((status)) && return "${status}"

  if ((timeout_seconds > 0)); then
    vllm_ci_monotonic_deciseconds now
    deadline=$((now + timeout_seconds * 10))
    while vllm_ci_process_is_live "${VLLM_CI_ACTIVE_PID}"; do
      vllm_ci_monotonic_deciseconds now
      ((now >= deadline)) && break
      sleep 0.1
    done
    if vllm_ci_process_is_live "${VLLM_CI_ACTIVE_PID}"; then
      timed_out=1
      vllm_ci_terminate_process_group \
        "${VLLM_CI_ACTIVE_PGID}" "${grace_seconds}" || true
    fi
  fi

  if ((timed_out)) \
    && vllm_ci_process_is_live "${VLLM_CI_ACTIVE_PID}"; then
    return 124
  fi
  wait "${VLLM_CI_ACTIVE_PID}" || status=$?
  if ! vllm_ci_terminate_process_group \
    "${VLLM_CI_ACTIVE_PGID}" "${grace_seconds}"; then
    ((status == 0)) && status=1
    return "${status}"
  fi
  VLLM_CI_ACTIVE_PID=""
  VLLM_CI_ACTIVE_PGID=""
  ((timed_out)) && return 124
  return "${status}"
}

vllm_ci_deadline_command_timeout() {
  local deadline=$1
  local maximum=$2
  local -n destination=$3
  local now=0
  local remaining_deciseconds=0

  vllm_ci_monotonic_deciseconds now
  remaining_deciseconds=$((deadline - now))
  ((remaining_deciseconds > 0)) || return 1
  destination=$(((remaining_deciseconds + 9) / 10))
  ((destination > maximum)) && destination=${maximum}
  return 0
}

vllm_ci_read_container_ids() {
  local -n destination=$1
  local timeout_seconds=$2
  local label_selector=$3
  local output=""

  destination=()
  output=$(vllm_ci_run_tracked_command "${timeout_seconds}" 1 \
    docker container ls -aq \
    --filter "label=${label_selector}") || return 1
  [[ -z "${output}" ]] || mapfile -t destination <<< "${output}"
}

vllm_ci_cleanup_labeled_containers() {
  local label_selector=$1
  local grace_seconds=$2
  local command_timeout_seconds=$3
  local cleanup_timeout_seconds=$4
  local required_empty_observations=${5:-1}
  local deadline=0
  local empty_observations=0
  local list_succeeded=0
  local now=0
  local operation_timeout=0
  local -a containers=()

  vllm_ci_monotonic_deciseconds now
  deadline=$((now + cleanup_timeout_seconds * 10))
  while vllm_ci_deadline_command_timeout \
    "${deadline}" "${command_timeout_seconds}" operation_timeout; do
    if ! vllm_ci_read_container_ids \
      containers "${operation_timeout}" "${label_selector}"; then
      echo "Unable to list Docker containers during teardown" >&2
      empty_observations=0
      sleep 0.2
      continue
    fi
    list_succeeded=1
    if ((${#containers[@]} == 0)); then
      empty_observations=$((empty_observations + 1))
      if ((empty_observations >= required_empty_observations)); then
        return 0
      fi
      sleep 0.2
      continue
    fi

    empty_observations=0
    echo "Gracefully stopping CI containers: ${containers[*]}"
    if vllm_ci_deadline_command_timeout \
      "${deadline}" "${command_timeout_seconds}" operation_timeout; then
      vllm_ci_run_tracked_command "${operation_timeout}" 1 \
        docker stop --time "${grace_seconds}" \
        "${containers[@]}" >/dev/null 2>&1 || true
    fi
    if vllm_ci_deadline_command_timeout \
      "${deadline}" "${command_timeout_seconds}" operation_timeout; then
      vllm_ci_run_tracked_command "${operation_timeout}" 1 \
        docker rm -f "${containers[@]}" >/dev/null 2>&1 || true
    fi
  done

  if ((list_succeeded == 0)); then
    echo "Unable to verify Docker container cleanup before its deadline" >&2
  elif ((${#containers[@]})); then
    echo "Docker containers remain after teardown: ${containers[*]}" >&2
  else
    echo "Unable to confirm stable Docker cleanup before its deadline" >&2
  fi
  return 1
}
