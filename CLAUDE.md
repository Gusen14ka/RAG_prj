# CLAUDE.md — RAG_prj

## Обзор проекта

Локальный RAG-pipeline для поиска по PDF-документам **без LLM** (retrieval-only).
Гибридный поиск: векторный (multilingual-e5-large) + BM25 + CrossEncoder reranker.
Два режима: `search` (поиск по документу) и `trainer` (генерация вопросов из чанков).

---

## Архитектура

```
PDF → parsing.py → recursive_chunking → chunks
                                          ├── embedding.py  (multilingual-e5-large, numpy dot-product)
                                          └── keyword_search.py (BM25Okapi + pymorphy3 + razdel)
                                                      └── rrf_scoring.py (Reciprocal Rank Fusion)
                                                                └── reranker.py (bge-reranker-v2-m3)
```

### Точки входа
- `main.py` — CLI (режимы `search` и `trainer`)
- `api.py` — FastAPI backend (порт 8000), используется в Docker
- `pipeline.py` — общая инициализация pipeline, используется обоими выше
- `docker-compose.yml` — backend (Python/uvicorn) + frontend (nginx, порт 80)

### Ключевые параметры (заданы в `pipeline.py`)
| Параметр | Значение | Назначение |
|---|---|---|
| `TOP_K_RETRIEVAL` | 5 | Топ-N из embedding и BM25 |
| `TOP_K_RERANK` | 3 | Финальных результатов после rerank |
| `RERANK_THRESHOLD` | 0.3 | Порог релевантности (ниже — отбрасываем) |
| `chunk_size_words` | 200 | Размер чанка в словах |
| `overlap_words` | 20 | Перекрытие чанков |

---

## Структура файлов

```
RAG_prj/
├── main.py                   — CLI точка входа
├── api.py                    — FastAPI backend
├── pipeline.py               — общая инициализация pipeline (пути, build_search_pipeline)
├── parsing.py                — PDF → чанки (pdfminer, normalize, clean)
├── embedding.py              — векторный поиск (multilingual-e5-large + numpy)
├── keyword_search.py         — BM25 поиск (pymorphy3 + razdel, ленивый MorphAnalyzer)
├── reranker.py               — CrossEncoder reranker (bge-reranker-v2-m3)
├── trainer_mode.py           — генерация вопросов (natasha + yake)
├── config/
│   ├── models_config.json    — список моделей для скачивания
│   └── requirements.txt      — зависимости (очищены от дублей и лишних пакетов)
├── scripts/
│   └── ensure_models.py      — скачивание моделей с HuggingFace
├── utils/
│   ├── recursive_chunking.py — рекурсивный чанкер
│   ├── rrf_scoring.py        — Reciprocal Rank Fusion
│   ├── load_chunks.py        — загрузка .jsonl / .json (единственная реализация)
│   └── __init__.py
├── frontend/index.html        — SPA на ванильном JS (тёмная/светлая тема)
├── nginx/
│   ├── Dockerfile             — nginx образ
│   └── nginx.conf             — прокси /api/ → backend:8000
├── Dockerfile                 — python:3.11-slim образ бэкенда
├── docker-compose.yml         — backend + frontend сервисы
├── models/                    — [не в git] локальные веса моделей (~5.3 GB)
├── data/                      — [не в git] кэш чанков и индексов
└── test.pdf                   — тестовый PDF-документ
```

---

## Запуск

### Локально
```bash
# 1. Установить зависимости
python3 -m venv .venv && source .venv/bin/activate
pip install -r config/requirements.txt

# 2. Скачать модели (~5.3 GB)
python3 scripts/ensure_models.py

# 3. Запустить поиск (из любой директории — пути абсолютные)
python3 main.py           # режим search (по умолчанию)
python3 main.py trainer   # режим генерации вопросов
```

### Docker
```bash
docker compose up --build   # ~25 GB на диске
docker compose down
```

### Сбросить кэш (новый PDF)
```bash
rm -rf data/
```

---

## Модели

| Модель | Задача | Размер |
|---|---|---|
| `intfloat/multilingual-e5-large` | Embedding (векторный поиск) | ~4.2 GB |
| `BAAI/bge-reranker-v2-m3` | CrossEncoder reranking | ~1.1 GB |

Путь локально: `models/multilingual-e5-large`, `models/bge-reranker-v2-m3`

---

## Технические детали

- **Почему numpy вместо FAISS**: FAISS конфликтует с CrossEncoder по OpenMP на macOS ARM.
- **BM25**: `pymorphy3` + `razdel` для лемматизации; `MorphAnalyzer` инициализируется лениво при первом вызове.
- **RRF**: `k=60` (стандарт), равные веса для embedding и BM25.
- **Trainer mode**: `natasha` + `yake` для генерации вопросов без LLM; `check_answer` делает один батч-вызов `model.predict()` вместо трёх.
- **Prefix embedding**: чанки → `"passage: "`, запросы → `"query: "` (требование multilingual-e5).
- **Пути**: все пути вычисляются через `Path(__file__).parent` в `pipeline.py` — проект запускается из любой директории.

---

## Исправленные проблемы (code review 2026-04-25)

| # | Что исправлено |
|---|---|
| 1 | Удалён `utils/group_pages_to_containers.py` — мёртвый файл (старая версия логики) |
| 2 | Убран дубликат `load_chunks` из `embedding.py`, везде используется `utils/load_chunks` |
| 3 | Переименован `faiss_id` → `array_idx` в `embedding.py` |
| 4 | Общая init-логика вынесена в `pipeline.py` — устранено дублирование `main.py`/`api.py` |
| 5 | В `trainer_mode.check_answer()` три `predict()` объединены в один батч-вызов |
| 6 | Голые `except:` заменены на `except Exception:` в `trainer_mode.py` |
| 7 | Очищен `requirements.txt`: убраны дубли `setuptools`/`razdel`, лишние `pymorphy2`/`pymorphy2-dicts-ru` |
| 8 | Удалены неиспользуемые функции `pdf_to_plain_text`, `save_plain_text`, `load_plain_text` из `parsing.py` |
| 9 | Удалена неиспользуемая `INLINE_HEADER_RE` из `trainer_mode.py` |
| 10 | Все пути переведены на `Path(__file__).parent` через `pipeline.py` |
| 11 | `MorphAnalyzer` инициализируется лениво (`_get_morph()`) в `keyword_search.py` |
| 12 | Исправлена опечатка `mathes` → `matches` в `trainer_mode._keyword_coverage()` |

---

## Полный анализ и оптимизация (2026-04-25)

### Маршрут запроса (frontend → ответ)

```
index.html  →  POST /search  →  asyncio.to_thread(perform_search)
    │
    ├── embedding_search()     — numpy dot-product, top-K=5
    ├── search_bm25()          — pymorphy3 + BM25Okapi, top-K=5
    ├── rrf_scoring()          — Reciprocal Rank Fusion (k=60), объединяем 2 списка
    └── rerank()               — CrossEncoder bge-reranker-v2-m3, top-K=3
            └── фильтр score >= RERANK_THRESHOLD (0.3)
```

### Где теряется время (типичный запрос)

| Этап | Время | Примечание |
|---|---|---|
| Embedding поиск | ~50–200 ms | Зависит от числа чанков, numpy dot-product |
| BM25 поиск | ~5–30 ms | Лемматизация через pymorphy3 |
| RRF | <1 ms | Чистая арифметика |
| Reranker | ~200–800 ms | CrossEncoder — самый тяжёлый этап |
| **Итого** | **~300–1000 ms** | Без кэша |

### Оптимизации

**LRU-кэш запросов (`api.py`)**
- `OrderedDict`-кэш на 64 записи, защищён `threading.Lock()`
- При повторном запросе возвращает результат мгновенно, всё тяжёлое вычисление пропускается
- Вывод в лог: `[SEARCH] cache hit: '...'`

**Тайминги в логах**
- `api.py`: каждый запрос логирует `embed=Xms bm25=Xms rrf=Xms rerank=Xms total=Xms`
- `pipeline.py`: холодный старт логирует время каждого из 4 этапов + суммарное `Cold start total: Xms`

### Исправления trainer_mode.py

**Корень проблемы: слишком мало вопросов (~2 вместо многих)**

| # | Проблема | Исправление |
|---|---|---|
| 1 | `clean_text()` переводил текст в нижний регистр перед natasha | Разделено: `clean_text_for_nlp()` (регистр сохранён) и `clean_text_for_check()` (lowercase только для сравнения ответов) |
| 2 | Типы `property`, `comparison`, `cause` не имели шаблонов вопросов — предложения классифицировались, но тихо выбрасывались | Добавлены шаблоны для всех 7 типов |
| 3 | Тип `usage` (использование/применение) полностью отсутствовал | Добавлен новый тип `usage` с 5 шаблонами, Q: "Для чего используется X?" |
| 4 | `"как"` в dependency-паттерне давал ложные срабатывания почти на каждое предложение | Удалён из списка лемм |
| 5 | `len(answer.split()) > 12` обрезал допустимые ответы | Лимит поднят до 20–22 слов |
| 6 | Порог `min_score=0.65` был слишком высок | Снижен до `0.55` |
| 7 | Нет диагностики — невозможно понять, на каком шаге отсеиваются предложения | Добавлен подробный stats-вывод: `sents= short= long= no_type= no_pattern= invalid= low_score= ok= types=` |

**Новые шаблоны определений** (было 4, стало 12):
- "X определяется как Y", "Под X понимают Y", "X обозначает Y"
- "X — это Y", "X представляет собой Y", "термином X называют Y"

### Улучшение parsing.py — удаление псевдокода

**Было**: только `while/for/if ... end while/for/if` блоки (Pascal-стиль с суффиксом).

**Стало** (`_remove_algo_blocks()` переписан как конечный автомат):

| Тип | Пример | Логика |
|---|---|---|
| while/for/if блоки | `while x do ... end while` | Прежняя логика, оставлена |
| Pascal begin/end | `begin ... end` | Новый `_PASCAL_BEGIN_RE` / `_PASCAL_END_RE` |
| Русский заголовок алгоритма | `Алгоритм 2.1:` | `_ALGO_HEADER_RU_RE` + флаг `skip_next_noncyrillic` |
| Блок некириллического кода | ≥4 строки без кириллицы + `:=`/`->`/`return`/… | `_CODE_SIGNS_RE` + буфер `noncyrillic_run` |

Некириллические строки накапливаются в буфер, который сбрасывается либо в `result` (если ≤3 строк или нет признаков кода), либо выбрасывается полностью (если ≥4 строк с признаками кода).

---

*Обновлён: 2026-04-25 — полный анализ и оптимизация завершены.*
