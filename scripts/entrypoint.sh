#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# Entrypoint for Django + Gunicorn on Render
# -------------------------------------------------------------

# TensorFlow / ML optimization envs (keep from your old script)
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
export TF_ENABLE_ONEDNN_OPTS=${TF_ENABLE_ONEDNN_OPTS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1}
export KMP_AFFINITY=${KMP_AFFINITY:-"granularity=fine,compact,1,0"}

# Gunicorn configuration (can override with env GUNICORN_CMD_ARGS)
## Increase default timeout: some predictions may take longer than 120s on limited CPUs.
## Allow max-requests to recycle workers and reduce memory growth over time.
## Use a single thread to avoid TensorFlow threading deadlocks inside worker processes
## Increase timeout further to allow longer-running inferences (but see notes below)
export GUNICORN_CMD_ARGS=${GUNICORN_CMD_ARGS:---workers=1 --threads=1 --timeout 600 --graceful-timeout=120 --max-requests=50 --max-requests-jitter=10 --log-level=info}

# -------------------------------------------------------------
# Confirm Render-assigned port
# -------------------------------------------------------------
if [ -z "${PORT:-}" ]; then
    echo "ERROR: Render did not assign PORT. Exiting."
    exit 1
fi
echo "Render assigned PORT=${PORT}"

# -------------------------------------------------------------
# Go to project root
# -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------------------------------------------------------------
# Run migrations & collect static files
# -------------------------------------------------------------
echo "Running Django migrations..."
python manage.py migrate --noinput || true

echo "Collecting static files..."
python manage.py collectstatic --noinput || true

# -------------------------------------------------------------
# Wait until port is free
# -------------------------------------------------------------
wait_for_port_free() {
    local port=$1
    local timeout=${2:-60}
    local start=$(date +%s)

    while :; do
        listener=$(lsof -tiTCP:${port} -sTCP:LISTEN || true)
        if [ -z "$listener" ]; then
            echo "Port ${port} is free."
            break
        fi
        now=$(date +%s)
        elapsed=$((now - start))
        if [ $elapsed -ge $timeout ]; then
            echo "Warning: Port ${port} still in use after ${timeout}s. Proceeding anyway..."
            break
        fi
        echo "Port ${port} is in use, waiting..."
        sleep 1
    done
}

wait_for_port_free "${PORT}"

# -------------------------------------------------------------
# Start Gunicorn in retry loop
# -------------------------------------------------------------
echo "Starting Gunicorn on 0.0.0.0:${PORT}..."
while :; do
    gunicorn PlantLeafDiseasePrediction.wsgi:application --bind 0.0.0.0:${PORT} ${GUNICORN_CMD_ARGS} && break
    echo "Gunicorn failed to start (bind error?), retrying in 2s..."
    sleep 2
done
