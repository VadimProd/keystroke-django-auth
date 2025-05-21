import random
import csv
import shutil

from pathlib import Path
from collections import OrderedDict

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import SetPasswordForm
from .models import KeystrokeSample
from django.conf import settings

MAX_ATTEMPTS = 2

CSV_PATH = Path(settings.BASE_DIR) / 'data' / 'keystroke_data.csv'

# При старте приложения удаляем папку 'data' и создаём заново
if CSV_PATH.parent.exists():
    shutil.rmtree(CSV_PATH.parent)

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

def parse_timing_data(timing_str):
    parts = timing_str.strip().split('\t')
    data = OrderedDict()
    for i in range(0, len(parts), 2):
        key = parts[i]
        value = parts[i + 1] if i + 1 < len(parts) else ''
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value
    return data

def read_header():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        return next(reader, [])

def write_header(columns):
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

def append_row(row, columns):
    row_list = [row.get(col, '') for col in columns]
    with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_list)

def save_csv(sample, timing_data):
    # Разбираем timing_data в OrderedDict, чтобы сохранить порядок
    timing_dict = parse_timing_data(timing_data)
    base_cols = ['user_id', 'typed_text', 'timestamp']
    timing_cols = list(timing_dict.keys())  # без сортировки

    columns = base_cols + timing_cols

    existing_cols = read_header()

    if not existing_cols:
        write_header(columns)
    else:
        existing_base = [col for col in existing_cols if col in base_cols]
        existing_timing = [col for col in existing_cols if col not in base_cols]
        new_timing = [col for col in timing_cols if col not in existing_timing]

        final_columns = existing_base + existing_timing + new_timing

        if final_columns != existing_cols:
            with open(CSV_PATH, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                old_rows = list(reader)

            write_header(final_columns)
            with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_columns)
                for row in old_rows:
                    for col in final_columns:
                        if col not in row:
                            row[col] = ''
                    writer.writerow(row)

        columns = final_columns

    # Записываем строку
    row = {
        'user_id': sample.user.id,
        'typed_text': sample.typed_text,
        'timestamp': sample.timestamp.isoformat(),
    }
    row.update(timing_dict)  # В правильном порядке
    append_row(row, columns)

def keystroke_input(request):
    if 'keystroke_attempt' not in request.session:
        request.session['keystroke_attempt'] = 1

    if request.method == "POST":
        form = SetPasswordForm(request.user, request.POST)
        typed_text = request.POST.get("typed_text", "").strip()
        timing_data = request.POST.get("timing_data", "").strip()
            
        # Сохраняем запись в БД
        sample = KeystrokeSample.objects.create(
            user=request.user,
            typed_text=typed_text,
            timing_data=timing_data
        )

        # save_csv(sample=sample, timing_data=timing_data)

        request.session['keystroke_attempt'] += 1
        if request.session['keystroke_attempt'] > MAX_ATTEMPTS:
            del request.session['keystroke_attempt']
            return redirect('logout')

    return render(request, "keystroke_input.html", {
        "attempt": request.session.get('keystroke_attempt', 1),
        "max_attempts": MAX_ATTEMPTS
    })


@login_required
def secret(request):
    confidence = random.uniform(80, 99)
    return render(request, "secret.html", {"confidence": confidence})
