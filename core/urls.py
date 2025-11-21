from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.index, name='index'),
    path('health/', views.health, name='health'),
    path('health/detail/', views.health_detail, name='health_detail'),
    path('api/initialize-model/', views.initialize_model_view, name='initialize_model'),
]
