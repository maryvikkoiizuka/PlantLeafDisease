# ml_model.py
"""
Machine Learning Model Integration for Plant Disease Prediction
This module handles the connection to the PlantDiseaseAI model
"""

import os
import sys
import threading
import numpy as np
from tensorflow import keras
from PIL import Image
import json

# Add the PlantDiseaseAI project to the path
PLANT_DISEASE_AI_PATH = r"c:\Users\admin\OneDrive - Auckland Institute of Studies\Desktop\PlantDiseaseAI"
if PLANT_DISEASE_AI_PATH not in sys.path:
    sys.path.insert(0, PLANT_DISEASE_AI_PATH)


class PlantDiseaseDetector:
    """
    Plant Disease Detection Model using CNN
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the model
        
        Args:
            model_path: Path to the saved model. If None, will look for a trained model.
        """
        self.model = None
        self.model_path = model_path
        self.class_indices = None
        self.image_size = (224, 224)  # Model expects 150x150 RGB images
        
    def load_model(self, model_path):
        """
        Load a pre-trained model
        
        Supports both formats:
        - .keras (modern Keras format - recommended)
        - .h5 (legacy HDF5 format)
        
        Args:
            model_path: Path to the saved model file (.keras or .h5)
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Keras automatically handles both .keras and .h5 formats
            self.model = keras.models.load_model(model_path)
            self.model_path = model_path

            # Try to detect model input size and adjust preprocessing
            try:
                input_shape = None
                # Common attribute
                if hasattr(self.model, 'input_shape') and self.model.input_shape is not None:
                    input_shape = self.model.input_shape

                # Some models expose `inputs` with TensorShape
                if input_shape is None and hasattr(self.model, 'inputs') and getattr(self.model, 'inputs'):
                    try:
                        shape_obj = self.model.inputs[0].shape
                        # TensorShape.as_list() may be available
                        if hasattr(shape_obj, 'as_list'):
                            input_shape = tuple(shape_obj.as_list())
                        else:
                            input_shape = tuple(shape_obj)
                    except Exception:
                        input_shape = None

                # Normalize and extract height/width for channels-last or channels-first
                if input_shape is not None:
                    # input_shape can be nested (for multi-input models)
                    if isinstance(input_shape, tuple) and len(input_shape) > 0 and isinstance(input_shape[0], tuple):
                        input_shape = input_shape[0]

                    # Typical TF shape: (None, H, W, C)
                    try:
                        if len(input_shape) >= 3:
                            # Remove batch dim if present
                            shape_list = list(input_shape)
                            if shape_list[0] is None:
                                shape_list = shape_list[1:]

                            # Channels-last e.g. [H, W, C]
                            if len(shape_list) == 3 and (shape_list[-1] == 1 or shape_list[-1] == 3):
                                h = int(shape_list[0])
                                w = int(shape_list[1])
                                self.image_size = (w, h)
                            # Channels-first e.g. [C, H, W]
                            elif len(shape_list) == 3 and (shape_list[0] == 1 or shape_list[0] == 3):
                                h = int(shape_list[1])
                                w = int(shape_list[2])
                                self.image_size = (w, h)
                    except Exception:
                        # If any parsing fails, keep default image_size
                        pass
            except Exception:
                # Non-fatal: leave default image_size
                pass

            # Determine format
            file_ext = os.path.splitext(model_path)[1]
            format_type = "Keras format (.keras)" if file_ext == ".keras" else "HDF5 format (.h5)"

            print(f"Model loaded successfully from {model_path}")
            print(f"Format: {format_type}")
            print(f"Using image_size={self.image_size} for preprocessing")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Supported formats: .keras (recommended) or .h5 (legacy)")
            return False
    
    def load_class_indices(self, json_path):
        """
        Load class indices mapping
        
        Args:
            json_path: Path to the class indices JSON file
        """
        try:
            with open(json_path, 'r') as f:
                self.class_indices = json.load(f)
            print(f"Class indices loaded successfully from {json_path}")
        except Exception as e:
            print(f"Error loading class indices: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.array: Preprocessed image array
        """
        try:
            # Open image
            image = Image.open(image_path)
            
            # Convert to RGB if not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size (150x150)
            image = image.resize(self.image_size)
            
            # Convert to array as float32
            image_array = np.array(image).astype('float32')

            # Try model-specific preprocessing if the model is known (e.g. EfficientNet)
            _preprocessed = None
            try:
                model_name = getattr(self.model, 'name', '').lower() if self.model is not None else ''
                layer_names = [l.name.lower() for l in self.model.layers[:6]] if self.model is not None else []

                # EfficientNet detection
                if 'efficientnet' in model_name or any('efficientnet' in n for n in layer_names):
                    try:
                        from tensorflow.keras.applications.efficientnet import preprocess_input as _eff_pre
                        _preprocessed = _eff_pre(image_array)
                    except Exception:
                        _preprocessed = None

                # You can add other backbone preprocess checks here (resnet, mobilenet, etc.)
            except Exception:
                _preprocessed = None

            if _preprocessed is None:
                # default normalization used during training for many models
                image_array = image_array / 255.0
            else:
                image_array = _preprocessed

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image_path):
        """
        Make a prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Prediction results with disease name and confidence
        """
        if self.model is None:
            return {"error": "Model not loaded. Please load a model first."}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            if processed_image is None:
                return {"error": "Failed to process image"}
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name (if indices are loaded)
            predicted_class_name = "Unknown"
            if self.class_indices:
                # Convert index to string to match JSON keys
                idx_str = str(predicted_class_idx)
                predicted_class_name = self.class_indices.get(idx_str, "Unknown")
            
            return {
                "disease": predicted_class_name,
                "confidence": confidence,
                "all_predictions": {
                    "class_index": int(predicted_class_idx),
                    "confidence_scores": predictions[0].tolist()
                }
            }
        
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        return results


# Global model instance + lock for thread-safe lazy initialization
_detector_instance = None
_detector_lock = threading.Lock()


def get_detector():
    """
    Thread-safe get-or-create for the global detector instance.

    Returns:
        PlantDiseaseDetector: The detector instance
    """
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = PlantDiseaseDetector()
    return _detector_instance


def initialize_model(model_path=None, class_indices_path=None):
    """
    Initialize the model with optional paths
    
    Args:
        model_path: Path to the saved Keras model
        class_indices_path: Path to class indices JSON file
        
    Returns:
        bool: True if initialization successful
    """
    detector = get_detector()
    
    if model_path and os.path.exists(model_path):
        detector.load_model(model_path)
    
    if class_indices_path and os.path.exists(class_indices_path):
        detector.load_class_indices(class_indices_path)
    
    return detector.model is not None
