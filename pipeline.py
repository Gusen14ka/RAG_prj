"""
Общая инициализация search pipeline.
Используется в main.py (CLI) и api.py (FastAPI).
"""
from pathlib import Path
import time

from parsing import pdf_to_chunks, save_chunks
from embedding import build_model, build_and_save_embeddings, load_embeddings
from keyword_search import create_bm25_index_pipeline, load_bm25
from utils.load_chunks import load_chunks, load_chunks_with_key
from reranker import build_reranker

# ── Пути (всегда относительно корня проекта, независимо от CWD) ──────────────
BASE_DIR            = Path(__file__).parent
DATA_DIR            = BASE_DIR / "data"
CHUNK_PATH          = DATA_DIR / "chunks.jsonl"
CHUNK_WITH_KEY_PATH = DATA_DIR / "chunks_with_key.json"
SOURCE_PDF_PATH     = BASE_DIR / "doc.pdf"
EMBEDDINGS_PATH     = DATA_DIR / "embeddings.npy"
BM25_PATH           = DATA_DIR / "bm25.pkl"
QUESTIONS_BANK_PATH = DATA_DIR / "questions_bank.jsonl"

# ── Гиперпараметры поиска ─────────────────────────────────────────────────────
TOP_K_RETRIEVAL  = 5
TOP_K_RERANK     = 3
RERANK_THRESHOLD = 0.3

# ── Гиперпараметры тренажёра ──────────────────────────────────────────────────
ANSWER_THRESHOLD  = 0.3
Q_SCORE_WEIGHT    = 0.35
SRC_SCORE_WEIGHT  = 0.45
GOLD_SCORE_WEIGHT = 0.20


def load_or_build_chunks():
    """
    Загружает чанки из кэша или строит из PDF.

    Логика (костыль против кривого парсинга):
    - Если оба файла есть → загрузить из кэша.
    - Если chunks.jsonl отсутствует → парсить PDF заново.
    - Если chunks.jsonl есть, но chunks_with_key.json нет → перечитать JSONL
      и пересохранить словарь (без повторного парсинга PDF).
    """
    DATA_DIR.mkdir(exist_ok=True)

    if not CHUNK_PATH.exists() or not CHUNK_WITH_KEY_PATH.exists():
        if not SOURCE_PDF_PATH.exists():
            raise FileNotFoundError(
                f"PDF не найден: {SOURCE_PDF_PATH}. "
                "Положите документ в корень проекта как test.pdf."
            )
        if not CHUNK_PATH.exists():
            chunks = pdf_to_chunks(str(SOURCE_PDF_PATH))
        else:
            chunks = load_chunks(str(CHUNK_PATH))
        save_chunks(chunks, str(CHUNK_PATH), str(CHUNK_WITH_KEY_PATH))
        print(f"    Создано {len(chunks)} чанков.")
    else:
        chunks = load_chunks(str(CHUNK_PATH))
        print(f"    Загружено {len(chunks)} чанков из кэша.")

    chunks_with_key = load_chunks_with_key(str(CHUNK_WITH_KEY_PATH))
    return chunks, chunks_with_key


def build_search_pipeline(verbose: bool = True) -> dict:
    """
    Инициализирует полный search pipeline.

    Возвращает словарь:
        chunks, chunks_with_key, embed_model, rerank_model, embeddings, bm25
    """
    t_total = time.perf_counter()

    if verbose:
        print("[1/4] Чанкинг PDF...")
        print("    Примечание: чтобы обработать новый PDF — удали папку data/")
    t0 = time.perf_counter()
    chunks, chunks_with_key = load_or_build_chunks()
    t_chunks = time.perf_counter()
    if verbose:
        print(f"    Чанкинг: {(t_chunks - t0) * 1000:.0f}ms")

    if verbose:
        print("[2/4] Загрузка моделей...")
    t0 = time.perf_counter()
    embed_model  = build_model()
    rerank_model = build_reranker()
    t_models = time.perf_counter()
    if verbose:
        print(f"    Модели загружены: {(t_models - t0) * 1000:.0f}ms")

    if verbose:
        print("[3/4] Векторный индекс (numpy)...")
    t0 = time.perf_counter()
    if not EMBEDDINGS_PATH.exists():
        if verbose:
            print("    Кодируем чанки...")
        embeddings = build_and_save_embeddings(chunks, embed_model, str(EMBEDDINGS_PATH))
    else:
        embeddings = load_embeddings(str(EMBEDDINGS_PATH))
    t_embed = time.perf_counter()
    if verbose:
        print(f"    Эмбеддингов: {embeddings.shape}  ({(t_embed - t0) * 1000:.0f}ms)")

    if verbose:
        print("[4/4] BM25 индекс...")
    t0 = time.perf_counter()
    if not BM25_PATH.exists():
        if verbose:
            print("    Строим BM25 индекс...")
        create_bm25_index_pipeline(str(CHUNK_PATH), str(BM25_PATH))
    bm25 = load_bm25(str(BM25_PATH))
    t_bm25 = time.perf_counter()
    if verbose:
        print(f"    BM25 загружен: {(t_bm25 - t0) * 1000:.0f}ms")
        print(f"    Cold start total: {(t_bm25 - t_total) * 1000:.0f}ms")

    return {
        "chunks":          chunks,
        "chunks_with_key": chunks_with_key,
        "embed_model":     embed_model,
        "rerank_model":    rerank_model,
        "embeddings":      embeddings,
        "bm25":            bm25,
    }
