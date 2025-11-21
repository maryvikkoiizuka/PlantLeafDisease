#!/bin/bash
set -e

# Simple Render deployment entrypoint
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1

echo "Starting application on port $PORT"

# Run migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

# Start Gunicorn
gunicorn PlantLeafDiseasePrediction.wsgi:application --bind 0.0.0.0:$PORT --workers=1 --threads=1 --timeout=600
