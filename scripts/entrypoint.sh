#!/bin/bash
set -e

# Render deployment entrypoint
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Starting application on port $PORT"

# Run migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

echo "Preloading TensorFlow and ML model (this may take 1-2 minutes on first run)..."

# Start Gunicorn with very extended timeout to allow TensorFlow + model loading
# Workers=1, threads=1 to minimize memory usage on free tier
gunicorn PlantLeafDiseasePrediction.wsgi:application \
  --bind 0.0.0.0:$PORT \
  --workers=1 \
  --threads=1 \
  --timeout=3600 \
  --graceful-timeout=120 \
  --keep-alive=75
