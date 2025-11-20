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
        """
        try:
            model_path = os.path.join(settings.BASE_DIR, 'models', 'plant_disease_model.keras')
            class_indices_path = os.path.join(settings.BASE_DIR, 'models', 'class_indices.json')

            # Prefer the modern .keras format, fallback to .h5
            if os.path.exists(model_path):
                initialize_model(model_path=model_path, class_indices_path=class_indices_path)
            else:
                alt_h5 = os.path.join(settings.BASE_DIR, 'models', 'plant_disease_model.h5')
                if os.path.exists(alt_h5):
                    initialize_model(model_path=alt_h5, class_indices_path=class_indices_path)
        except Exception as e:
            logging.getLogger(__name__).exception('Failed to preload ML model: %s', e)
