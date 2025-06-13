from django.contrib import admin
from django.urls import path, include
from accounts import views as accounts_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', accounts_views.home, name='home'),  # главная страница
    path('accounts/', include('accounts.urls')),
    path('keystroke/', include('keystroke.urls')),
]