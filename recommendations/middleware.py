from django.utils.deprecation import MiddlewareMixin

class LogRequestMiddleware(MiddlewareMixin):
    def process_request(self, request):
        print(f"[HF Space] {request.method} {request.path} from {request.META.get('REMOTE_ADDR')}")
        return None
