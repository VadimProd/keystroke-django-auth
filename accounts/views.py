import joblib
import io
import json

from scipy.special import expit 

from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.contrib import messages
from keystroke.models import KeystrokeLogin, KeystrokeProfile
from django.contrib.auth.models import User
from django.core.files.storage import default_storage

def home(request):
    return render(request, "home.html")

def keystroke_input(request):
    return render(request, "keystroke_input.html")

def logout_view(request):
    logout(request)
    return redirect('home')

def extract_features_from_timing(timing_str):
    parts = timing_str.split('\t')
    features = []
    for i in range(1, len(parts), 2):
        try:
            value = float(parts[i])
        except (ValueError, IndexError):
            value = 0.0
        features.append(value)
    return features

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()

            # Получаем timing_data из POST
            typed_text = request.POST.get('username')
            timing_data = request.POST.get('timing_data')

            # Загружаем обученную модель пользователя
            try:
                profile = KeystrokeProfile.objects.get(user=user)
                if not profile.model_file:
                    messages.error(request, 'Biometric model not found for this user.')
                    return render(request, 'login.html', {'form': form})

                with default_storage.open(profile.model_file.name, 'rb') as f:
                    data = joblib.load(f)  # Загружаем словарь {'model': ..., 'scaler': ...}
                    model = data['model']
                    scaler = data['scaler']

                features = extract_features_from_timing(timing_data)
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                decision_scores = model.decision_function(features_scaled)
                probabilities = expit(decision_scores)

                status = "Принято (настоящий пользователь)" if prediction == 1 else "Отклонено (возможный злоумышленник)"
                print(f"Оценка аномалии = {decision_scores}, вероятность легитимности = {probabilities} — {status}")
                print(timing_data)

                if prediction == 1:
                    login(request, user)  # Успешный вход
                    # KeystrokeLogin.objects.create(
                    #     user=user,
                    #     typed_text=typed_text,
                    #     timing_data=timing_data
                    # )
                    messages.success(request, 'Login successful with biometric verification.')
                    return redirect('home')
                else:
                    messages.error(request, 'Biometric verification failed. Access denied.')
                    return render(request, 'login.html', {'form': form})

            except KeystrokeProfile.DoesNotExist:
                messages.error(request, 'No biometric profile found for this user.')
                return render(request, 'login.html', {'form': form})

        else:
            messages.error(request, 'Invalid username or password.')

    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})

# def login_view(request):
#     if request.method == 'POST':
#         form = AuthenticationForm(request, data=request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             login(request, user)  # вызываем login с другим именем (auth_login), чтобы не путать с функцией

#             # Получаем данные из формы
#             typed_text = request.POST.get('username')  # Или можно request.POST.get('typed_text'), если будет
#             timing_data = request.POST.get('timing_data')

#             # Сохраняем keystroke данные в БД
#             KeystrokeLogin.objects.create(
#                 user=user,
#                 typed_text=typed_text,
#                 timing_data=timing_data
#             )

#             messages.info(request, 'Successful login.')
#             return redirect('home')
#         else:
#             messages.error(request, 'Invalid username or password.')
#     else:
#         form = AuthenticationForm()

#     return render(request, 'login.html', {'form': form})

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