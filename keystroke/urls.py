from django.urls import path
from . import views

urlpatterns = [
    path('input/', views.keystroke_input, name='keystroke_input'),
    path('secret/', views.secret, name='secret'),
]