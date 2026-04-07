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
├── main.py                        — точка входа, основной pipeline
├── parsing.py                     — PDF → чанки
├── embedding.py                   — векторный поиск (numpy)
├── keyword_search.py              — BM25 поиск
├── reranker.py                    — CrossEncoder переранжирование
├── config/
│   ├── models_config.json         — список моделей для скачивания
│   └── requirements.txt           — зависимости
├── scripts/
│   └── ensure_models.py           — скачивание моделей с HuggingFace
├── utils/
│   ├── recursive_chunking.py      — рекурсивный чанкер
│   ├── rrf_scoring.py             — Reciprocal Rank Fusion
│   ├── load_chunks.py             — загрузка .jsonl / .json
│   └── group_pages_to_containers.py
├── models/                        — локальные модели (не в git)
├── data/                          — кэш чанков и индексов (не в git)
└── test.pdf                       — тестовый документ
```

---

## Требования

- Python 3.10+
- ~2.5 GB свободного места (модели)
- ~4 GB RAM

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

## Скачивание моделей

Модели скачиваются один раз (~2.3 GB):

```bash
python3 scripts/ensure_models.py   # macOS / Linux
python  scripts/ensure_models.py   # Windows
```

Модели сохраняются в `models/` и не попадают в git.

> Если получаете предупреждение `unauthenticated requests to HF Hub` — скорость ограничена.
> Для ускорения: `export HF_TOKEN=hf_ваш_токен` (токен бесплатно на huggingface.co).

---

## Запуск

Положите свой PDF как `test.pdf` в корень проекта (или используйте уже имеющийся).

```bash
python3 main.py   # macOS / Linux
python  main.py   # Windows
```

> **Важно:** запускать из корневой директории проекта.
> Если хотите обработать другой PDF — удалите папку `data/` перед запуском.

### Ожидаемый вывод

```
[1/5] Чанкинг PDF...
    Создано 14 чанков.
[2/5] Загрузка моделей...
    Модели загружены.
[3/5] Векторный индекс (numpy)...
    Эмбеддингов: (14, 1024)
[4/5] BM25 индекс...
    BM25 загружен.

[5/5] Система готова. Введите 'выход' для завершения.

Введите запрос: парадокс Рассела

────────────────────────────────────────────────────────────
Топ-2 результатов для: «парадокс Рассела»
────────────────────────────────────────────────────────────

[1] chunk_id: 1.1.3:1  score=0.9800
Раздел: Множества и отношения
Подраздел: Парадокс Рассела
...
```

---

## Используемые модели

| Модель | Задача | Размер |
|--------|--------|--------|
| [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | Embedding (векторный поиск) | ~1.2 GB |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Reranking (CrossEncoder) | ~1.1 GB |

---

## Зависимости

| Пакет | Назначение |
|-------|------------|
| `pdfminer.six` | Извлечение текста из PDF |
| `sentence-transformers` | Загрузка и инференс моделей |
| `numpy` | Векторный поиск (dot-product) |
| `rank-bm25` | BM25 индекс |
| `pymorphy3` + `razdel` | Лемматизация русского текста |
| `torch` | Backend для трансформеров |

---

## Платформы

| Платформа | Статус |
|-----------|--------|
| macOS ARM (M1/M2/M3) | ✅ Протестировано |
| macOS x86 | ✅ Должно работать |
| Linux | ✅ Должно работать |
| Windows 10/11 | ✅ Должно работать |

> На macOS используется numpy вместо faiss во избежание конфликта OpenMP между torch и faiss.

---

## Изменения относительно ветки main

### Критические баги — исправлены

| Файл | Проблема | Исправление |
|------|----------|-------------|
| `reranker.py:24` | Сортировка по `x[0]` (chunk_id) вместо `x[1]` (score) — результаты выдавались в случайном порядке | `key=lambda x: x[1]` |
| `reranker.py:8` | Неверная аннотация возвращаемого типа `List[str]` | `List[Tuple[str, float]]` |
| `parsing.py:273` | `save_chunks(chunks, output_file)` — передавался только 1 путь вместо 2 | добавлен `output_file_with_key` |
| `parsing.py:12` | `HEADER_RE = pattern = re.compile(...)` — лишний псевдоним `pattern` | `HEADER_RE = re.compile(...)` |
| `parsing.py:41` | Дублирующий `import re` в середине файла | удалён |
| `parsing.py:160` | `m.group(1).strip()` без проверки — `group(1)` опциональна, `None.strip()` → `AttributeError` | добавлена проверка `m.group(1) is not None` |
| `keyword_search.py:65,75,86` | Пути с `\\` (Windows-стиль) — не работали на macOS/Linux | заменены на `/` |
| `embedding.py:95` | `build_model()` вызывался внутри `search()` — модель перезагружалась при каждом запросе | модель передаётся как параметр |
| `reranker.py:10` | `CrossEncoder` создавался внутри `rerank()` — модель перезагружалась при каждом запросе | добавлен `build_reranker()`, модель передаётся снаружи |
| `scripts/ensure_models.py:25` | `open("config/models_config.json")` без `encoding` — падал на Windows | добавлен `encoding="utf-8"` |
| `config/requirements.txt` | Файл в кодировке UTF-16 — `pip install` не мог его прочитать | перезаписан в UTF-8 |

### Архитектурные улучшения

| Файл | Что изменено |
|------|-------------|
| `embedding.py` | `faiss` заменён на `numpy` dot-product — устранён segfault (конфликт OpenMP faiss + torch на macOS ARM) |
| `embedding.py` | Функции переименованы: `build_and_save_faiss_index` → `build_and_save_embeddings`, `load_index_and_metadata` → `load_embeddings` |
| `keyword_search.py` | Добавлена `search_bm25(bm25, chunks, query)` — поиск без I/O по готовым объектам |
| `keyword_search.py` | `save_index` больше не сохраняет `bm25_tokenized` — он не нужен для поиска |
| `reranker.py` | Добавлена `build_reranker()` — явная точка инициализации модели |
| `scripts/ensure_models.py` | Пути через `pathlib.Path` — кроссплатформенно; `model_path.mkdir(parents=True)` |
| `main.py` | Полный рефакторинг: модели загружаются **один раз** при старте, затем переиспользуются в цикле |
| `main.py` | Добавлен фильтр по порогу `RERANK_THRESHOLD = 0.3` — нерелевантные чанки не показываются |
| `main.py` | Добавлена проверка наличия `test.pdf` перед парсингом |
| `main.py` | `os.makedirs("data", exist_ok=True)` — папка `data/` создаётся автоматически |
| `main.py` | UTF-8 вывод на Windows (`sys.stdout.reconfigure`) |
| `main.py` | Хардкод запроса заменён на `input()` с циклом и командой выхода |

### Порядок загрузки (критично для macOS ARM)

В `main.py` модели (`build_model`, `build_reranker`) загружаются **до** любых операций с numpy-индексами. Это предотвращает конфликт потоков OpenMP между torch и faiss/numpy на Apple Silicon.
