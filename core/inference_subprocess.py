#!/usr/bin/env python
"""Lightweight inference runner executed as a short-lived subprocess.

Usage: python inference_subprocess.py <image_path> [model_path] [class_indices_path]

This script loads the model, runs prediction on the given image, and
prints a single JSON object to stdout with the result or an error.
"""
import sys
import os
import json
from pathlib import Path
import time


def _write_log(entry: str):
    try:
        repo_root = Path(__file__).resolve().parents[1]
        log_path = repo_root / 'render_errors.log'
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"=== {ts} (subprocess) ===\n")
            f.write(entry)
            f.write('\n\n')
    except Exception:
        pass

def main():
    try:
        image_path = sys.argv[1]
    except IndexError:
        print(json.dumps({'error': 'image_path required'}))
        sys.exit(2)

    model_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    class_indices_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

    # Defer heavy imports to runtime so the script is lightweight at import time
    try:
        _write_log('subprocess: importing TF/PIL')
        from tensorflow import keras
        import numpy as np
        from PIL import Image
        _write_log('subprocess: imported TF/PIL successfully')
    except Exception as e:
        _write_log(f'subprocess: failed to import TF/PIL: {str(e)}')
        print(json.dumps({'error': f'Failed to import TF/PIL: {str(e)}'}))
        sys.exit(3)

    # Reuse the main project's PlantDiseaseDetector to ensure identical preprocessing
    try:
        # Import from project package so preprocessing/model logic matches main code
        from core.ml_model import PlantDiseaseDetector
        detector = PlantDiseaseDetector()
    except Exception as e:
        _write_log(f'subprocess: failed to import PlantDiseaseDetector: {str(e)}')
        print(json.dumps({'error': f'Failed to import detector implementation: {str(e)}'}))
        sys.exit(6)

    try:
        if model_path and os.path.exists(model_path):
            _write_log(f'subprocess: loading model from {model_path}')
            detector.load_model(model_path)
            _write_log('subprocess: model loaded successfully')
        else:
            _write_log(f'subprocess: model_path not found or not provided: {model_path}')
    except Exception as e:
        _write_log(f'subprocess: Failed to load model: {str(e)}')
        print(json.dumps({'error': f'Failed to load model: {str(e)}'}))
        sys.exit(4)

    try:
        if class_indices_path and os.path.exists(class_indices_path):
            detector.load_class_indices(class_indices_path)
    except Exception:
        pass

    try:
        _write_log(f'subprocess: starting prediction for {image_path}')
        res = detector.predict(image_path)
        _write_log(f'subprocess: prediction finished: {str(res)[:1000]}')
        print(json.dumps(res))
        sys.exit(0)
    except Exception as e:
        _write_log(f'subprocess: Prediction failed: {str(e)}')
        print(json.dumps({'error': f'Prediction failed: {str(e)}'}))
        sys.exit(5)


if __name__ == '__main__':
    main()
