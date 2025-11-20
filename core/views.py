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
from .ml_model import get_detector, initialize_model, predict_via_worker, get_inference_pool, predict_via_subprocess
import time

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


                # Run inference in a short-lived subprocess to avoid worker deadlocks.
                # This is more reliable in constrained environments; if you need
                # lower latency we can re-introduce the worker-pool as an opt-in.
                try:
                    _write_error_log('PREDICTION START: calling predict_via_subprocess()')
                    t0 = time.time()
                except Exception:
                    t0 = time.time()

                # Try to pass configured model path found on the in-process detector
                try:
                    detector = get_detector()
                    model_path = getattr(detector, 'model_path', None)
                except Exception:
                    model_path = None

                prediction = predict_via_subprocess(temp_path, timeout=300, model_path=model_path)

                try:
                    t1 = time.time()
                    _write_error_log(f'PREDICTION END: elapsed_seconds={t1 - t0:.3f}')
                except Exception:
                    logger.exception('Failed to write post-prediction log')

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

        # Also ensure the worker pool is initialized with the same model paths
        try:
            # This will lazily start a single-worker pool that preloads the model
            get_inference_pool(model_path=model_path, class_indices_path=class_indices_path)
        except Exception:
            logger.exception('Failed to initialize inference worker pool')
        
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


@require_http_methods(["GET"])
def debug_render_errors(request):
    """Return the tail of the server-side `render_errors.log` for debugging.

    This endpoint is intentionally guarded by `settings.DEBUG` to avoid
    exposing logs in production. It returns JSON with `log_exists` and
    `tail` (last ~4000 chars) when available.
    """
    if not settings.DEBUG:
        return JsonResponse({'status': 'forbidden', 'message': 'Debug endpoint disabled'}, status=403)

    log_path = os.path.join(settings.BASE_DIR, 'render_errors.log')
    if not os.path.exists(log_path):
        return JsonResponse({'status': 'ok', 'log_exists': False, 'tail': ''})

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tail = content[-4000:] if len(content) > 4000 else content
        return JsonResponse({'status': 'ok', 'log_exists': True, 'tail': tail})
    except Exception as e:
        logger.exception('Failed to read render_errors.log')
        return JsonResponse({'status': 'error', 'message': 'Could not read log file'}, status=500)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def ping(request):
    """Lightweight test endpoint to verify POST reaches Django without running prediction.

    Use this to distinguish platform-level 502s from model-processing errors.
    It's CSRF-exempt so it can be called from curl without cookies.
    """
    try:
        cl = request.META.get('CONTENT_LENGTH')
        return JsonResponse({
            'status': 'ok',
            'method': request.method,
            'content_length': cl,
        })
    except Exception as e:
        logger.exception('Error in ping endpoint')
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
