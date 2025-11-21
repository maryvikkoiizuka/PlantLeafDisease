from django.apps import AppConfig
import os
import logging
from django.conf import settings

from .ml_model import initialize_model


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        """
        Preload the ML model once when the Django app registry is ready.

        This attempts to initialize the model from the project's `models/`
        directory if the files are present. Any errors are logged and
        do not prevent Django from starting.
        
        Uses the CNN simple model by default to minimize memory usage,
        falls back to EfficientNet or H5 format if not available.
        """
        try:
            models_dir = os.path.join(settings.BASE_DIR, 'models')
            
            # Try CNN simple first (smaller, ~32MB vs 49MB for EfficientNet)
            model_path = os.path.join(models_dir, 'plant_disease_model_cnn_simple.keras')
            class_indices_path = os.path.join(models_dir, 'class_indices_cnn_simple.json')
            
            if os.path.exists(model_path) and os.path.exists(class_indices_path):
                logging.getLogger(__name__).info(f"Loading CNN simple model from {model_path}")
                initialize_model(model_path=model_path, class_indices_path=class_indices_path)
                return
            
            # Fallback to EfficientNet keras
            model_path = os.path.join(models_dir, 'plant_disease_model_efficientnetb0.keras')
            class_indices_path = os.path.join(models_dir, 'class_indices_efficientnetb0.json')
            
            if os.path.exists(model_path) and os.path.exists(class_indices_path):
                logging.getLogger(__name__).info(f"Loading EfficientNet model from {model_path}")
                initialize_model(model_path=model_path, class_indices_path=class_indices_path)
                return
            
            # Fallback to default keras
            model_path = os.path.join(models_dir, 'plant_disease_model.keras')
            class_indices_path = os.path.join(models_dir, 'class_indices.json')
            
            if os.path.exists(model_path):
                logging.getLogger(__name__).info(f"Loading default model from {model_path}")
                initialize_model(model_path=model_path, class_indices_path=class_indices_path)
                return
            
            # Fallback to H5 format
            model_path = os.path.join(models_dir, 'plant_disease_model.h5')
            class_indices_path = os.path.join(models_dir, 'class_indices.json')
            
            if os.path.exists(model_path):
                logging.getLogger(__name__).info(f"Loading H5 model from {model_path}")
                initialize_model(model_path=model_path, class_indices_path=class_indices_path)
                return
            
            logging.getLogger(__name__).warning("No model files found in models/ directory")
            
        except Exception as e:
            logging.getLogger(__name__).exception('Failed to preload ML model: %s', e)
