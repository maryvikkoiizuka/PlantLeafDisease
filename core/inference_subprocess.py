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
        from tensorflow import keras
        import numpy as np
        from PIL import Image
    except Exception as e:
        print(json.dumps({'error': f'Failed to import TF/PIL: {str(e)}'}))
        sys.exit(3)

    # Simple local detector implementation mirroring PlantDiseaseDetector
    class _LocalDetector:
        def __init__(self):
            self.model = None
            self.class_indices = None
            self.image_size = (224, 224)

        def load_model(self, p):
            self.model = keras.models.load_model(p)
            # try to infer input size
            try:
                inp = None
                if hasattr(self.model, 'input_shape') and self.model.input_shape is not None:
                    inp = self.model.input_shape
                if inp is None and hasattr(self.model, 'inputs') and getattr(self.model, 'inputs'):
                    try:
                        shape_obj = self.model.inputs[0].shape
                        if hasattr(shape_obj, 'as_list'):
                            inp = tuple(shape_obj.as_list())
                        else:
                            inp = tuple(shape_obj)
                    except Exception:
                        inp = None
                if inp is not None:
                    if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], tuple):
                        inp = inp[0]
                    if len(inp) >= 3:
                        sl = list(inp)
                        if sl[0] is None:
                            sl = sl[1:]
                        if len(sl) == 3 and (sl[-1] == 1 or sl[-1] == 3):
                            h = int(sl[0]); w = int(sl[1]); self.image_size = (w,h)
                        elif len(sl) == 3 and (sl[0] == 1 or sl[0] == 3):
                            h = int(sl[1]); w = int(sl[2]); self.image_size = (w,h)

        def load_class_indices(self, jp):
            try:
                with open(jp, 'r') as f:
                    self.class_indices = json.load(f)
            except Exception:
                pass

        def preprocess(self, image_path):
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(self.image_size)
            arr = np.array(image).astype('float32') / 255.0
            arr = np.expand_dims(arr, axis=0)
            return arr

        def predict(self, image_path):
            if self.model is None:
                return {'error': 'Model not loaded'}
            x = self.preprocess(image_path)
            preds = self.model.predict(x, verbose=0)
            idx = int(preds[0].argmax())
            conf = float(preds[0][idx])
            name = 'Unknown'
            if self.class_indices:
                name = self.class_indices.get(str(idx), name)
            return {'disease': name, 'confidence': conf, 'class_index': idx}

    detector = _LocalDetector()

    try:
        if model_path and os.path.exists(model_path):
            detector.load_model(model_path)
    except Exception as e:
        print(json.dumps({'error': f'Failed to load model: {str(e)}'}))
        sys.exit(4)

    try:
        if class_indices_path and os.path.exists(class_indices_path):
            detector.load_class_indices(class_indices_path)
    except Exception:
        pass

    try:
        res = detector.predict(image_path)
        print(json.dumps(res))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({'error': f'Prediction failed: {str(e)}'}))
        sys.exit(5)


if __name__ == '__main__':
    main()
