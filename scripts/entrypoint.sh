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

# Run any pending migrations (safe to ignore failures in some environments)
echo "Running migrations..."
python manage.py migrate --noinput || true

# Collect static files (no-op if not configured)
echo "Collecting static files..."
python manage.py collectstatic --noinput || true

PORT=${PORT:-8000}
echo "Starting Gunicorn on 0.0.0.0:${PORT} with args: ${GUNICORN_CMD_ARGS}"

exec gunicorn PlantLeafDiseasePrediction.wsgi:application --bind 0.0.0.0:${PORT} ${GUNICORN_CMD_ARGS}
