from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.index, name='index'),
    path('health/', views.health, name='health'),
    path('api/initialize-model/', views.initialize_model_view, name='initialize_model'),
    path('debug/render-errors/', views.debug_render_errors, name='debug_render_errors'),
    path('api/ping/', views.ping, name='ping'),
]
