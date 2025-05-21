from django.urls import path
from . import views
from keystroke import views as keystroke_views
from django.contrib.auth.views import LoginView

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('keystroke_input/', keystroke_views.keystroke_input, name='keystroke_input'),
]