#!/usr/bin/env bash
set -euo pipefail

# Entrypoint for Render / containerless deployments
# Sets TensorFlow-related environment variables to reduce memory/CPU pressure
# then runs migrations, collects static files, and starts Gunicorn with
# a conservative worker/thread config to avoid OOM on small instances.

export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
export TF_ENABLE_ONEDNN_OPTS=${TF_ENABLE_ONEDNN_OPTS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1}
export KMP_AFFINITY=${KMP_AFFINITY:-"granularity=fine,compact,1,0"}

# Allow override of Gunicorn args via env var; otherwise keep minimal workers
export GUNICORN_CMD_ARGS=${GUNICORN_CMD_ARGS:---workers=1 --threads=2 --timeout 120}

echo "Starting entrypoint: TF envs set. TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL}, OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# Ensure we run from repository root (script may be executed from ./scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Run any pending migrations (safe to ignore failures in some environments)
echo "Running migrations... (cwd=${REPO_ROOT})"
python manage.py migrate --noinput || true

# Collect static files (no-op if not configured)
echo "Collecting static files... (cwd=${REPO_ROOT})"
python manage.py collectstatic --noinput || true

PORT=${PORT:-8000}
echo "Preparing to start server on 0.0.0.0:${PORT} with args: ${GUNICORN_CMD_ARGS}"

# Helper: check whether the port is already in use and try to free it.
free_port_if_needed() {
	local port=$1
	# Try to find a pid listening on the TCP port using ss (works on most Linux hosts)
	if command -v ss >/dev/null 2>&1; then
		pid=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$0 ~ p { if(match($0, /pid=[0-9]+/)) { m=substr($0,RSTART,RLENGTH); sub(/pid=/, "", m); print m; exit } }') || true
	else
		pid=""
	fi

	if [ -n "${pid}" ]; then
		echo "Port ${port} is currently used by pid ${pid}. Attempting to stop it..."
		# Try graceful termination
		kill -TERM "${pid}" 2>/dev/null || true
		# Wait up to 5 seconds for process to exit
		for i in 1 2 3 4 5; do
			sleep 1
			if ! ss -ltnp 2>/dev/null | grep -q ":${port} "; then
				echo "Port ${port} freed after SIGTERM"
				return 0
			fi
		done
		echo "Process ${pid} did not exit; sending SIGKILL..."
		kill -KILL "${pid}" 2>/dev/null || true
		sleep 1
		if ss -ltnp 2>/dev/null | grep -q ":${port} "; then
			echo "Warning: port ${port} is still in use after SIGKILL; aborting start to avoid conflicts."
			return 1
		fi
		echo "Port ${port} freed after SIGKILL"
	fi
	return 0
}

# Ensure the port is free (preempt conflicting processes)
if ! free_port_if_needed "${PORT}"; then
	echo "Failed to free port ${PORT}; exiting to avoid duplicate listeners."
	exit 1
fi

# If gunicorn is available use it (production). Otherwise fall back to Django runserver (useful for local Windows/Git Bash testing).
if command -v gunicorn >/dev/null 2>&1; then
	echo "Starting Gunicorn on 0.0.0.0:${PORT}"
	exec gunicorn PlantLeafDiseasePrediction.wsgi:application --bind 0.0.0.0:${PORT} ${GUNICORN_CMD_ARGS}
else
	echo "gunicorn not found, falling back to 'python manage.py runserver' (for local testing only)"
	exec python manage.py runserver 0.0.0.0:${PORT}
fi
