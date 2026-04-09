FROM python:3.11-slim

WORKDIR /app

# Загруаем системные зависимости
# Базовые build-essential и libgomp1 для питоновских библиотек - библиотека OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Отдельно копируем только requirements, тк редко меняется
COPY config/requirements.txt config/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r config/requirements.txt

# Копируем исходный код
COPY . .

# Декларируем, что будем слушать порт 8000
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
