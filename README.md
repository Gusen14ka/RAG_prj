# RAG Pipeline — Retrieval-Augmented Generation (retrieval-only)

Локальный поисковый pipeline по PDF-документу без обращения к внешним API.
Гибридный поиск: векторный (multilingual-e5-large) + ключевые слова (BM25) + переранжирование (bge-reranker-v2-m3).

---

## Как это работает

```
PDF
 └─► Parsing (pdfminer)          — извлечение текста, разбивка на разделы
      └─► Chunking (recursive)   — рекурсивная нарезка на чанки по 200 слов
           ├─► Embedding search  — косинусное similarity через numpy (multilingual-e5-large)
           └─► Keyword search    — BM25 с лемматизацией (pymorphy3 + razdel)
                └─► RRF Fusion   — объединение результатов (Reciprocal Rank Fusion)
                     └─► Rerank  — финальный CrossEncoder (bge-reranker-v2-m3)
```

**Шаг 1 — Parsing.**
`pdfminer.six` извлекает текст постранично. Страницы группируются по нумерованным разделам (`1.1`, `1.2.3` и т.д.) в контейнеры. Текст очищается: убираются PDF-артефакты `(cid:...)`, номера страниц, формулы заменяются токеном `<FORMULA>`.

**Шаг 2 — Chunking.**
Каждый контейнер-раздел нарезается рекурсивно: сначала по абзацам (`\n\n`), затем по строкам, предложениям и т.д. Размер чанка — 200 слов, перекрытие — 20 слов.

**Шаг 3 — Embedding.**
`multilingual-e5-large` кодирует чанки с префиксом `"passage: "`, запрос — с `"query: "`. Поиск через матричное dot-product (cosine similarity на нормализованных векторах).

**Шаг 4 — BM25.**
Текст лемматизируется через `pymorphy3` + `razdel`. Индекс строится через `rank-bm25 (BM25Okapi)`.

**Шаг 5 — RRF.**
Результаты двух поисков объединяются через Reciprocal Rank Fusion с равными весами.

**Шаг 6 — Reranker.**
`bge-reranker-v2-m3` (CrossEncoder) переоценивает финальные кандидаты попарно `(query, chunk)`. Результаты ниже порога `0.3` отбрасываются.

---

## Структура проекта

```
RAG_prj/
├── main.py                        — CLI: режимы "тренажёр" и "учебник"
├── api.py                         — FastAPI backend для Docker / веб-UI
├── parsing.py                     — PDF → контейнеры страниц → чанки
├── embedding.py                   — сборка/поиск эмбеддингов (numpy + sentence-transformers)
├── keyword_search.py              — BM25 поиск (rank-bm25)
├── reranker.py                    — CrossEncoder / reranker
├── docker-compose.yml             — Docker сервисы: backend + nginx frontend
├── Dockerfile                     — образ бэкенда с Python и моделями
├── nginx/
│   ├── Dockerfile                 — образ frontend на nginx
│   └── nginx.conf                 — прокси на backend
├── config/
│   ├── models_config.json         — список моделей для загрузки
│   └── requirements.txt           — Python-зависимости (указаны в config/requirements.txt)
├── scripts/
│   └── ensure_models.py           — скачивание моделей (HuggingFace, локально)
├── utils/
│   ├── recursive_chunking.py      — рекурсивный чанкер
│   ├── rrf_scoring.py             — Reciprocal Rank Fusion
│   ├── load.py                    — функции загрузки JSON / JSONL и чаргов
│   └── save.py                    — функции сохранения чанков и индексов
├── models/                        — локальные модели (внешние веса не в репозитории)
├── data/                          — кэш чанков, эмбеддингов и индексов (не в git)
└── frontend/index.html            — минимальный UI
```

---

## Требования

- Python 3.10+
- Свободное место: минимально несколько гигабайт для моделей и кэшей; для полноценной работы с Docker образами рекомендуется иметь ~25 GB свободного места
- RAM: рекомендуется ≥4 GB (чем больше — тем комфортнее при инференсе)
- GPU: опционально. Проект может использовать CUDA для LLM и/или эмбеддингов, но это не обязательно.

---

## Установка

### macOS / Linux

```bash
git clone https://github.com/Gusen14ka/RAG_prj
cd RAG_prj
python3 -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
```

### Windows

```bat
git clone https://github.com/Gusen14ka/RAG_prj
cd RAG_prj
python -m venv .venv
.venv\Scripts\activate
pip install -r config/requirements.txt
```

---

## Docker

Проект можно запускать через Docker Compose: backend на Python + nginx frontend.

```bash
docker compose up --build
```

или, если используется старая версия Docker:

```bash
docker-compose up --build
```

> Важно: образы и контейнеры могут занимать до 25 GB на диске. Перед запуском проверьте свободное место.

Чтобы остановить и удалить контейнеры:

```bash
docker compose down
```

Чтобы удалить образ и освободить пространство:

```bash
docker system prune -a
```

---

## Скачивание моделей

Скрипт `scripts/ensure_models.py` загружает модели, указанные в `config/models_config.json`, в папку `models/`.

```bash
python3 scripts/ensure_models.py   # macOS / Linux
python  scripts/ensure_models.py   # Windows
```

Модели сохраняются в `models/` и не попадают в git.

Если появляется предупреждение про `unauthenticated requests to HF Hub`, скорость может быть ограничена. Для ускорения можно установить переменную окружения `HF_TOKEN` с вашим токеном HuggingFace.

---

## Запуск

Положите свой PDF как `data/source.pdf` или укажите путь в `config/paths.py` (по умолчанию используется `data/source.pdf`).

### Локально (CLI)

```bash
python3 main.py   # macOS / Linux
python  main.py   # Windows
```

CLI имеет два режима:
- `Тренажёр` — показывает случайные вопросы из банка и позволяет проверять ответы;
- `Учебник` — интерактивный поиск: с LLM (генерация ответа из найденных контекстов) и без LLM (только релевантные чанки).

### В Docker

```bash
docker compose up --build
```

Важно запускать из корневой директории проекта. Если вы хотите заново построить чанки/индексы для другого PDF — удалите содержимое `data/` и перезапустите.

### Ожидаемый вывод (пример)

```
Created 14 chunks.
Loading LLM model...
LLM susscesfully loaded
Embedding model will use CPU
Encoding chunks...
Loaded 14 chunks from cache.
System ready. Введите запрос.
```

---

## Используемые модели

Модели и их назначения указываются в `config/models_config.json`. В репозитории в папке `models/` ожидаются локальные копии моделей (скрипт `scripts/ensure_models.py` загружает их).

- `multilingual-e5-*` — модель эмбеддингов (используется для векторного поиска через нормализованные эмбеддинги).
- `bge-reranker*` или аналогичный cross-encoder — используется для финального переранжирования кандидатов.
- `qwen2.5-3b` (или другая LLM) — используется как локальная LLM для генерации ответов и переформулировки запросов (в `llm.py` реализована загрузка модели с возможностью 4-bit квантования для экономии памяти).

---

## Зависимости

Зависимости перечислены в `config/requirements.txt`. Ключевые пакеты:

- `pdfminer.six` — извлечение текста из PDF;
- `sentence-transformers` / `transformers` — загрузка и инференс моделей эмбеддингов и reranker;
- `numpy` — операции с эмбеддингами (dot-product / cosine similarity);
- `rank-bm25` — BM25 индекс для keyword search;
- `pymorphy3` + `razdel` — лемматизация и токенизация русского текста;
- `torch` — исполнение моделей; опционально с CUDA для ускорения инференса.

---

## Платформы

| Платформа | Статус |
|-----------|--------|
| macOS ARM (M1/M2/M3) | ✅ Протестировано |
| macOS x86 | ✅ Должно работать |
| Linux | ✅ Должно работать |
| Windows 10/11 | ✅ Должно работать |
