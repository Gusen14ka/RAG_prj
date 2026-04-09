FROM nginx:alpine

# Удаляем дефолтный кофиг из образа, тк он будет конфликтовать с нашим
RUN rm /etc/nginx/conf.d/default.conf

COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Копируем фронт в стандартную папку для статики nginx.
# /usr/share/nginx/html — это путь, который мы указали в root внутри nginx.conf.
COPY frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80

# CMD не нужен — базовый образ nginx:alpine уже содержит правильную команду запуска.
# Она запускает nginx в foreground-режиме (daemon off), что правильно для Docker:
# контейнер живёт пока жив главный процесс.