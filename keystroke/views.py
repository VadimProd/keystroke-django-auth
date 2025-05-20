from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import KeystrokeSample
import random  # для имитации вероятности

@login_required
def keystroke_input(request):
    if request.method == "POST":
        typed_text = request.POST.get("typed_text")
        timing_data = request.POST.get("timing_data")

        sample = KeystrokeSample(
            user=request.user,
            typed_text=typed_text,
            timing_data=timing_data
        )
        sample.save()
        return redirect('secret')

    return render(request, "keystroke_input.html")

@login_required
def secret(request):
    # На этом этапе будет производиться реальный анализ, а пока имитация:
    confidence = random.uniform(80, 99)  # случайная вероятность
    return render(request, "secret.html", {"confidence": confidence})