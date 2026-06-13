#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-/tmp/bci-runtime-logs}"
RUNTIME_ROLE="${RUNTIME_ROLE:-score}"
ALGORITHM_ADDRESS="${ALGORITHM_ADDRESS:-localhost:9981}"

export PYTHONPATH="${ROOT_DIR}/app/Algorithm:${ROOT_DIR}/app/CentralController:${ROOT_DIR}/app/Collector:${ROOT_DIR}/app/ProcessHub:${PYTHONPATH:-}"
export ALGORITHM_ADDRESS

mkdir -p "${RUNTIME_LOG_DIR}"

prepare_task_data_mount() {
  local virtual_receiver_data_dir="${ROOT_DIR}/app/Collector/Collector/receiver/virtual_receiver/data"
  if [[ -d "${TASK_DATA_DIR:-}" ]]; then
    rm -rf "${virtual_receiver_data_dir}"
    mkdir -p "$(dirname "${virtual_receiver_data_dir}")"
    ln -s "${TASK_DATA_DIR}" "${virtual_receiver_data_dir}"
    echo "[INFO] mounted TASK_DATA_DIR to virtual receiver data: ${TASK_DATA_DIR}"
  else
    echo "[WARN] TASK_DATA_DIR is missing or not a directory: ${TASK_DATA_DIR:-}"
    echo "[WARN] container will use any data already present in ${virtual_receiver_data_dir}"
  fi
}

validate_task_data_mount() {
  local virtual_receiver_data_dir="${ROOT_DIR}/app/Collector/Collector/receiver/virtual_receiver/data"
  local source_dat_file=""
  local target_dat_file=""

  if [[ -z "${TASK_DATA_DIR:-}" ]]; then
    echo "[ERROR] TASK_DATA_DIR is missing; mount the task data directory before starting the container" >&2
    return 1
  fi

  if [[ ! -d "${TASK_DATA_DIR}" ]]; then
    echo "[ERROR] TASK_DATA_DIR is missing or not a directory: ${TASK_DATA_DIR}" >&2
    return 1
  fi

  source_dat_file="$(find "${TASK_DATA_DIR}" -type f -name '*.dat' -print -quit 2>/dev/null || true)"
  if [[ -z "${source_dat_file}" ]]; then
    echo "[ERROR] TASK_DATA_DIR contains no .dat files: ${TASK_DATA_DIR}" >&2
    return 1
  fi
  echo "[INFO] TASK_DATA_DIR contains .dat files: ${source_dat_file}"

  if [[ ! -d "${virtual_receiver_data_dir}" ]]; then
    echo "[ERROR] virtual receiver data directory is missing: ${virtual_receiver_data_dir}" >&2
    return 1
  fi

  target_dat_file="$(find -L "${virtual_receiver_data_dir}" -type f -name '*.dat' -print -quit 2>/dev/null || true)"
  if [[ -z "${target_dat_file}" ]]; then
    echo "[ERROR] virtual receiver data directory contains no .dat files: ${virtual_receiver_data_dir}" >&2
    return 1
  fi
  echo "[INFO] virtual receiver data directory contains .dat files: ${target_dat_file}"
}

cleanup() {
  if [[ -n "${CENTROL_PID:-}" ]]; then kill "${CENTROL_PID}" 2>/dev/null || true; fi
  if [[ -n "${COLLECTOR_JAR_PID:-}" ]]; then kill "${COLLECTOR_JAR_PID}" 2>/dev/null || true; fi
  if [[ -n "${TASK_JAR_PID:-}" ]]; then kill "${TASK_JAR_PID}" 2>/dev/null || true; fi
  if [[ -n "${CENTRAL_PID:-}" ]]; then kill "${CENTRAL_PID}" 2>/dev/null || true; fi
  if [[ -n "${RUNTIME_STAGE_PID:-}" ]]; then kill "${RUNTIME_STAGE_PID}" 2>/dev/null || true; fi
  if [[ -n "${ALGORITHM_PID:-}" ]]; then kill "${ALGORITHM_PID}" 2>/dev/null || true; fi
  if [[ -n "${COLLECTOR_PID:-}" ]]; then kill "${COLLECTOR_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

run_bg() {
  local label="$1"
  local workdir="$2"
  shift 2
  echo "[INFO] starting ${label}" >&2
  (
    cd "${workdir}"
    "$@"
  ) >"${RUNTIME_LOG_DIR}/${label}.log" 2>&1 &
  echo $!
}

wait_for_tcp_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local timeout_seconds="${4:-60}"
  local waited_seconds=0
  while ! : <"/dev/tcp/${host}/${port}" 2>/dev/null; do
    if (( waited_seconds >= timeout_seconds )); then
      echo "[ERROR] timeout waiting for ${label} on ${host}:${port}" >&2
      return 1
    fi
    sleep 1
    waited_seconds=$((waited_seconds + 1))
  done
  echo "[INFO] ${label} is ready on ${host}:${port}"
}

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] RUNTIME_ROLE=${RUNTIME_ROLE}"
echo "[INFO] ALGORITHM_ADDRESS=${ALGORITHM_ADDRESS}"
echo "[INFO] TASK_DATA_DIR=${TASK_DATA_DIR:-}"
echo "[INFO] LEADERBOARD_TASK_DIR=${LEADERBOARD_TASK_DIR:-}"
echo "[INFO] PRELIMINARY_DATA_DIR=${PRELIMINARY_DATA_DIR:-}"

prepare_task_data_mount
validate_task_data_mount

CENTROL_PID="$(run_bg centrol "${ROOT_DIR}/proceed/centrol" java -jar centrol.jar)"
sleep "${CENTROL_START_WAIT_SECONDS:-15}"

CENTRAL_PID="$(run_bg central_python "${ROOT_DIR}/app/CentralController" "${PYTHON_BIN}" -u -m ApplicationFramework.main)"

export LAUNCHER_CONFIG_PATH="${ROOT_DIR}/app/ProcessHub/ApplicationFramework/config/RuntimeStageCoordinatorLauncherConfig.yml"
RUNTIME_STAGE_PID="$(run_bg runtime_stage "${ROOT_DIR}/app/ProcessHub" "${PYTHON_BIN}" -u -m ApplicationFramework.main)"
unset LAUNCHER_CONFIG_PATH

COLLECTOR_JAR_PID="$(run_bg collector_jar "${ROOT_DIR}/proceed/collector" java -jar collector.jar)"
TASK_JAR_PID="$(run_bg task_jar "${ROOT_DIR}/proceed/task" java -jar task.jar)"

wait_for_tcp_port 127.0.0.1 9002 collector_jar "${COLLECTOR_JAR_START_WAIT_SECONDS:-60}"
wait_for_tcp_port 127.0.0.1 9003 task_jar "${TASK_JAR_START_WAIT_SECONDS:-60}"

if [[ "${RUNTIME_ROLE}" != "score" && "${START_LOCAL_ALGORITHM:-0}" == "1" ]]; then
  ALGORITHM_PID="$(run_bg algorithm_python "${ROOT_DIR}/app/Algorithm" "${PYTHON_BIN}" -u -m Algorithm.main)"
  sleep "${ALGORITHM_START_WAIT_SECONDS:-15}"
else
  echo "[INFO] score mode: local Algorithm is not started"
fi

COLLECTOR_PID="$(run_bg collector_python "${ROOT_DIR}/app/Collector" "${PYTHON_BIN}" -u -m ApplicationFramework.main)"

echo "[INFO] starting processhub_python in foreground"
cd "${ROOT_DIR}/app/ProcessHub"
exec "${PYTHON_BIN}" -u -m ApplicationFramework.main
