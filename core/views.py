from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import os
import json
import logging
import traceback
from datetime import datetime
from .ml_model import get_detector, initialize_model

# Module logger
logger = logging.getLogger(__name__)

def _write_error_log(extra_info: str):
    try:
        log_path = os.path.join(settings.BASE_DIR, 'render_errors.log')
        ts = datetime.utcnow().isoformat() + 'Z'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"=== {ts} ===\n")
            f.write(extra_info)
            f.write('\n\n')
    except Exception:
        # Best-effort logging: do not raise
        logger.exception('Failed to write render_errors.log')


def health(request):
    """Readiness endpoint: returns 200 only when the ML model is loaded.

    Returns JSON with `model_loaded: true|false`. If the model is not yet
    loaded the endpoint returns HTTP 503 so load balancers know to wait.
    """
    detector = get_detector()
    model_loaded = detector.model is not None
    if model_loaded:
        return JsonResponse({"status": "ok", "model_loaded": True})
    else:
        return JsonResponse({"status": "starting", "model_loaded": False}, status=503)


@require_http_methods(["GET", "POST"])
def index(request):
    """
    Home page view for plant leaf disease prediction.
    Handles both GET requests (displaying the form) and POST requests (file upload).
    """
    if request.method == 'POST':
        try:
            # Handle file upload
            if 'image' in request.FILES:
                uploaded_file = request.FILES['image']

                # Save uploaded file temporarily
                temp_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
                os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

                with open(temp_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # Get prediction from ML model
                detector = get_detector()

                if detector.model is None:
                    return JsonResponse({
                        'success': False,
                        'error': 'ML model not loaded. Please initialize the model first.'
                    })

                prediction = detector.predict(temp_path)

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if 'error' in prediction:
                    return JsonResponse({
                        'success': False,
                        'error': prediction['error']
                    })

                return JsonResponse({
                    'success': True,
                    'predicted_class': prediction['disease'],
                    'confidence': prediction['confidence'] * 100,  # Convert to percentage
                    'message': f"Detected: {prediction['disease']} (Confidence: {prediction['confidence']:.2%})"
                })
            else:
                return JsonResponse({'success': False, 'error': 'No file provided'})
        except Exception as e:
            # Build detailed diagnostic info for logs (do not expose sensitive data to client)
            tb = traceback.format_exc()
            try:
                meta_info = {
                    'path': request.path,
                    'method': request.method,
                    'content_length': request.META.get('CONTENT_LENGTH'),
                    'remote_addr': request.META.get('REMOTE_ADDR'),
                    'user_agent': request.META.get('HTTP_USER_AGENT')
                }
            except Exception:
                meta_info = {'path': request.path}

            extra = f"Exception: {str(e)}\nMeta: {json.dumps(meta_info)}\nTraceback:\n{tb}"
            logger.exception('Unhandled error in index view: %s', str(e))
            _write_error_log(extra)

            # Return a safe JSON error for the client
            return JsonResponse({
                'success': False,
                'error': 'Server error while processing the image. Details logged.'
            }, status=500)
    
    return render(request, 'index.html')


@csrf_exempt
@require_http_methods(["POST"])
def initialize_model_view(request):
    """
    Initialize the ML model with provided paths
    Expects JSON data with model_path and class_indices_path
    """
    try:
        data = json.loads(request.body)
        model_path = data.get('model_path')
        class_indices_path = data.get('class_indices_path')
        
        if not model_path:
            return JsonResponse({
                'status': 'error',
                'message': 'model_path is required'
            })
        
        detector = get_detector()
        
        # Load model
        if os.path.exists(model_path):
            detector.load_model(model_path)
        else:
            return JsonResponse({
                'status': 'error',
                'message': f'Model file not found: {model_path}'
            })
        
        # Load class indices if provided
        if class_indices_path and os.path.exists(class_indices_path):
            detector.load_class_indices(class_indices_path)
        
        if detector.model is not None:
            return JsonResponse({
                'status': 'success',
                'message': 'Model initialized successfully'
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to load model'
            })
    
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error: {str(e)}'
        })
