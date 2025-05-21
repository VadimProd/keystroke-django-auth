from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.contrib import messages
from keystroke.models import KeystrokeLogin
from django.contrib.auth.models import User

def home(request):
    return render(request, "home.html")

def keystroke_input(request):
    return render(request, "keystroke_input.html")

def logout_view(request):
    logout(request)
    return redirect('home')

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)  # вызываем login с другим именем (auth_login), чтобы не путать с функцией

            # Получаем данные из формы
            typed_text = request.POST.get('username')  # Или можно request.POST.get('typed_text'), если будет
            timing_data = request.POST.get('timing_data')

            # Сохраняем keystroke данные в БД
            KeystrokeLogin.objects.create(
                user=user,
                typed_text=typed_text,
                timing_data=timing_data
            )

            messages.info(request, 'Successful login.')
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # здесь сохраняется и пароль
            login(request, user)  # сразу логиним пользователя
            return redirect('keystroke_input')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

# def register(request):
#     if request.method == "POST":
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('keystroke_input')  # сразу к сбору данных
#     else:
#         form = UserCreationForm()
#     return render(request, 'register.html', {'form': form})