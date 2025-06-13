# Dockerfile
FROM python:3.11-slim

# Установка зависимостей
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create working dir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]