from django.apps import AppConfig


class RecommendationsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommendations'
    label = 'recommendations_app'  # custom label to avoid conflicts
    
    def ready(self):
        from . import mlModel
        mlModel.load_model()
