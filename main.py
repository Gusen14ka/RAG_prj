import os
import sys
from pathlib import Path

# UTF-8 вывод на Windows (cmd/PowerShell по умолчанию cp1251)
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure') and hasattr(sys.stderr, 'reconfigure'):
        sys.stdout.reconfigure(encoding="utf-8") #type: ignore
        sys.stderr.reconfigure(encoding="utf-8") #type: ignore

from parsing import pdf_to_chunks, save_chunks
from embedding import (
    load_chunks,
    build_model,
    build_and_save_embeddings,
    load_embeddings,
    search as embedding_search,
)
from keyword_search import (
    create_bm25_index_pipeline,
    load_bm25,
    search_bm25,
)
from utils.rrf_scoring import rrf_scoring
from utils.load_chunks import load_chunks_with_key
from reranker import rerank, build_reranker

# ── Пути ────────────────────────────────────────────────────────────────
CHUNK_PATH          = "data/chunks.jsonl"
CHUNK_WITH_KEY_PATH = "data/chunks_with_key.json"
SOURCE_PDF_PATH     = "test.pdf"
EMBEDDINGS_PATH     = "data/embeddings.npy"
BM25_PATH           = "data/bm25.pkl"

TOP_K_RETRIEVAL  = 5
TOP_K_RERANK     = 3
RERANK_THRESHOLD = 0.3   # чанки ниже этого score не показываем


def main():
    os.makedirs("data", exist_ok=True)

    # ── 1. Чанкинг ──────────────────────────────────────────────────────
    print("[1/5] Чанкинг PDF...")
    print("    Примечание: чтобы обработать новый PDF — удали папку data/")
    if not Path(CHUNK_PATH).exists() or not Path(CHUNK_WITH_KEY_PATH).exists():
        if not Path(SOURCE_PDF_PATH).exists():
            print(f"    ОШИБКА: {SOURCE_PDF_PATH} не найден.")
            return
        chunks = pdf_to_chunks(SOURCE_PDF_PATH)
        save_chunks(chunks, CHUNK_PATH, CHUNK_WITH_KEY_PATH)
        print(f"    Создано {len(chunks)} чанков.")
    else:
        chunks = load_chunks(CHUNK_PATH)
        print(f"    Загружено {len(chunks)} чанков из кэша.")
    chunks_with_key = load_chunks_with_key(CHUNK_WITH_KEY_PATH)

    # ── 2. Загрузка моделей (до любых индексов) ──────────────────────────
    print("[2/5] Загрузка моделей...")
    embed_model  = build_model()
    rerank_model = build_reranker()
    print("    Модели загружены.")

    # ── 3. Векторные эмбеддинги (numpy, без faiss) ───────────────────────
    print("[3/5] Векторный индекс (numpy)...")
    if not Path(EMBEDDINGS_PATH).exists():
        print("    Кодируем чанки...")
        embeddings = build_and_save_embeddings(chunks, embed_model, EMBEDDINGS_PATH)
    else:
        embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"    Эмбеддингов: {embeddings.shape}")

    # ── 4. BM25 индекс ───────────────────────────────────────────────────
    print("[4/5] BM25 индекс...")
    if not Path(BM25_PATH).exists():
        print("    Строим BM25 индекс...")
        create_bm25_index_pipeline(CHUNK_PATH, BM25_PATH)
    bm25 = load_bm25(BM25_PATH)
    print("    BM25 загружен.")

    # ── 5. Цикл запросов ────────────────────────────────────────────────
    print("\n[5/5] Система готова. Введите 'выход' для завершения.\n")
    while True:
        query = input("Введите запрос: ").strip()
        if not query:
            continue
        if query.lower() in ("выход", "exit", "quit"):
            print("Выход.")
            break

        res_embedding = embedding_search(embeddings, chunks, query,
                                         model=embed_model, top_k=TOP_K_RETRIEVAL)
        res_keyword   = search_bm25(bm25, chunks, query, top_k=TOP_K_RETRIEVAL)

        res_rrf_ids  = rrf_scoring([res_embedding, res_keyword], weight_list=[1, 1])
        res_rrf      = [chunks_with_key[x[0]] for x in res_rrf_ids]

        res_rerank_ids = rerank(query, res_rrf, TOP_K_RERANK, model=rerank_model)

        # Фильтр по порогу: убираем нерелевантные чанки
        filtered = [(cid, score) for cid, score in res_rerank_ids
                    if score >= RERANK_THRESHOLD]

        if not filtered:
            print("\n    Ничего не найдено выше порога релевантности.")
            print()
            continue

        print(f"\n{'─'*60}")
        print(f"Топ-{len(filtered)} результатов для: «{query}»")
        print(f"{'─'*60}")
        for i, (cid, score) in enumerate(filtered, 1):
            ch   = chunks_with_key[cid]
            text = ch.get("raw_text", "").strip()
            print(f"\n[{i}] chunk_id: {cid}  score={score:.4f}")
            print(text[:400])
            if len(text) > 400:
                print("    ...")
        print()


if __name__ == "__main__":
    main()