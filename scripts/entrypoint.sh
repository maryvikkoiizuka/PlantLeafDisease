#!/bin/bash
set -e

# Render deployment entrypoint
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

echo "Starting application on port $PORT"

# Run migrations
echo "Running migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Pre-warm the model to avoid cold-start timeout issues
echo "Pre-warming ML model (this may take 1-2 minutes)..."
python manage.py warmup_model || echo "Model warm-up skipped or failed, will load on first request"

echo "Starting Gunicorn server..."

# Start Gunicorn with very extended timeout to allow initial TensorFlow loading
gunicorn PlantLeafDiseasePrediction.wsgi:application \
  --bind 0.0.0.0:$PORT \
  --workers=1 \
  --threads=1 \
  --timeout=3600 \
  --graceful-timeout=120 \
  --keep-alive=75 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
