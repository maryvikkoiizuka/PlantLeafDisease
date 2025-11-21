"""
Management command to pre-warm the ML model by running a dummy prediction.
This helps avoid timeout issues on the first real prediction.
"""
from django.core.management.base import BaseCommand
from django.conf import settings
import os
import logging
from core.ml_model import get_detector, initialize_model
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Pre-warm the ML model to avoid cold start timeouts'

    def handle(self, *args, **options):
        self.stdout.write('Warming up ML model...')
        
        try:
            detector = get_detector()
            
            if detector.model is None:
                self.stdout.write(self.style.WARNING('Model not loaded, attempting to load...'))
                model_path = os.path.join(settings.BASE_DIR, 'models', 'plant_disease_model.keras')
                class_indices_path = os.path.join(settings.BASE_DIR, 'models', 'class_indices.json')
                
                if os.path.exists(model_path):
                    initialize_model(model_path=model_path, class_indices_path=class_indices_path)
                else:
                    alt_h5 = os.path.join(settings.BASE_DIR, 'models', 'plant_disease_model.h5')
                    if os.path.exists(alt_h5):
                        initialize_model(model_path=alt_h5, class_indices_path=class_indices_path)
            
            if detector.model is None:
                self.stdout.write(self.style.ERROR('Failed to load model'))
                return
            
            # Run a dummy prediction to warm up the model
            self.stdout.write('Running dummy prediction to warm up model...')
            
            # Create a simple test image
            dummy_image = Image.new('RGB', (224, 224), color='green')
            dummy_path = os.path.join(settings.MEDIA_ROOT, 'warmup_test.jpg')
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            dummy_image.save(dummy_path)
            
            # Run prediction
            result = detector.predict(dummy_path)
            
            # Clean up
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
            
            self.stdout.write(self.style.SUCCESS('Model warm-up complete!'))
            self.stdout.write(f'Warm-up result: {result}')
            
        except Exception as e:
            logger.exception('Error during model warm-up')
            self.stdout.write(self.style.ERROR(f'Warm-up failed: {str(e)}'))
