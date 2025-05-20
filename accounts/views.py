from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth import logout

def home(request):
    print("Rendering home")
    return render(request, "home.html")

def logout_view(request):
    logout(request)
    return redirect('home')

def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, "register.html", {"form": form})