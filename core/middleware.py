import time
import os
import json
from django.conf import settings

def _write_error_log(extra_info: str):
    try:
        log_path = os.path.join(settings.BASE_DIR, 'render_errors.log')
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"=== {ts} ===\n")
            f.write(extra_info)
            f.write('\n\n')
    except Exception:
        # best-effort
        pass


class RequestLoggerMiddleware:
    """Middleware that logs incoming requests at the earliest point.

    This helps determine whether requests reach Django at all (before
    multipart parsing / view code runs). It logs minimal info: method,
    path, content-length, and client addr.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            meta = {
                'method': request.method,
                'path': request.path,
                'content_length': request.META.get('CONTENT_LENGTH'),
                'remote_addr': request.META.get('REMOTE_ADDR'),
                'user_agent': request.META.get('HTTP_USER_AGENT')
            }
            _write_error_log('MIDDLEWARE REQUEST ARRIVAL: ' + json.dumps(meta))
        except Exception:
            pass

        response = self.get_response(request)
        return response
