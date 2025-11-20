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

# Acquire a filesystem lock so only one process in this container performs
# migrations/collectstatic and starts the server. This prevents race
# conditions where multiple start commands try to bind the same PORT.
LOCK_FILE=/tmp/app_start.lock
exec 9>"${LOCK_FILE}"
echo "Acquiring start lock ${LOCK_FILE} (this will block if another process is starting)..."
# Block until we can acquire the exclusive lock on fd 9. The FD is kept
# open across exec so that the lock remains held for the lifetime of the
# server process that replaces this script via exec.
flock -x 9

# Now we hold the lock. Run any pending migrations (safe to ignore failures in some environments)
echo "Running migrations... (cwd=${REPO_ROOT})"
python manage.py migrate --noinput || true

# Collect static files (no-op if not configured)
echo "Collecting static files... (cwd=${REPO_ROOT})"
python manage.py collectstatic --noinput || true

PORT=${PORT:-8000}
echo "Preparing to start server on 0.0.0.0:${PORT} with args: ${GUNICORN_CMD_ARGS}"

# Diagnostic dump: capture listeners and processes so we can debug port conflicts.
dump_diagnostics() {
	echo "--- Network listeners (ss -ltnp) ---"
	if command -v ss >/dev/null 2>&1; then
		ss -ltnp || true
	else
		echo "ss not available"
	fi

	echo "--- Fallback netstat (netstat -tunlp) ---"
	if command -v netstat >/dev/null 2>&1; then
		netstat -tunlp || true
	else
		echo "netstat not available"
	fi

	echo "--- lsof listeners (lsof -i -n -P) ---"
	if command -v lsof >/dev/null 2>&1; then
		lsof -i -n -P | head -n 200 || true
	else
		echo "lsof not available"
	fi

	echo "--- ps snapshot (ps aux | head) ---"
	if command -v ps >/dev/null 2>&1; then
		ps aux | head -n 200 || true
	else
		echo "ps not available"
	fi
	echo "--- end diagnostics ---"
}

dump_diagnostics

# Log port owner if present, but do NOT try to free/kill it — let the
# hosting platform (Render) manage port assignments and lifecycle.
log_port_owner() {
	local port=$1
	pid=""
	if command -v ss >/dev/null 2>&1; then
		pid=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$0 ~ p { if(match($0, /pid=[0-9]+/)) { m=substr($0,RSTART,RLENGTH); sub(/pid=/, "", m); print m; exit } }') || true
	fi
	if [ -z "${pid}" ]; then
		if command -v lsof >/dev/null 2>&1; then
			pid=$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null | head -n1 || true)
		fi
	fi
	if [ -n "${pid}" ]; then
		owner_cmd=""
		if [ -r "/proc/${pid}/cmdline" ]; then
			owner_cmd=$(tr '\0' ' ' < /proc/${pid}/cmdline || true)
		fi
		echo "Note: port ${port} is currently used by pid ${pid}. Command: ${owner_cmd}. Not attempting to free it; letting Render manage lifecycle."
	else
		echo "No existing listener found on port ${port}."
	fi
}

# Record current owner but do not kill it; Render controls lifecycle
log_port_owner "${PORT}"

# If gunicorn is available use it (production). Otherwise fall back to Django runserver (useful for local Windows/Git Bash testing).
AUTO_INSTALL_GUNICORN=${AUTO_INSTALL_GUNICORN:-0}
if command -v gunicorn >/dev/null 2>&1; then
	echo "gunicorn found: $(gunicorn --version 2>&1 | head -n1)"
	echo "Starting Gunicorn on 0.0.0.0:${PORT}"
	# Dump diagnostics just before exec to capture the immediate state
	dump_diagnostics || true
	exec gunicorn PlantLeafDiseasePrediction.wsgi:application --bind 0.0.0.0:${PORT} ${GUNICORN_CMD_ARGS}
else
	echo "gunicorn not found in PATH. Checking installed Python packages..."
	if python -m pip show gunicorn >/dev/null 2>&1; then
		echo "gunicorn is installed in the Python environment (pip), but not on PATH. Showing 'pip show':"
		python -m pip show gunicorn || true
	else
		echo "gunicorn not installed in Python environment. Showing top of 'pip freeze' to help debug:" 
		python -m pip freeze | head -n 50 || true
	fi

	if [ "${AUTO_INSTALL_GUNICORN}" = "1" ]; then
		echo "AUTO_INSTALL_GUNICORN=1: attempting to install gunicorn via pip..."
		python -m pip install --no-cache-dir "gunicorn==20.1.0" || true
		if command -v gunicorn >/dev/null 2>&1; then
			echo "gunicorn installed; starting gunicorn now"
			dump_diagnostics || true
			exec gunicorn PlantLeafDiseasePrediction.wsgi:application --bind 0.0.0.0:${PORT} ${GUNICORN_CMD_ARGS}
		else
			echo "gunicorn still not found after installation attempt; falling back to runserver"
		fi
	else
		echo "Note: to auto-install gunicorn at startup set ENV AUTO_INSTALL_GUNICORN=1 (not recommended for production builds)."
	fi

	echo "Falling back to Django development server for now"
	exec python manage.py runserver 0.0.0.0:${PORT}
fi
