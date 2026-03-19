# embed_and_index.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from tqdm import tqdm
import os

# -------------------------
# Конфиг (подправь по необходимости)
MODEL_NAME = "models/multilingual-e5-large"  # компактная, хорошая для англ. + многие языки
INDEX_PATH = "data/faiss_index.ivf"   # файл для FAISS
METADATA_PATH = "data/chunks_metadata.json"
EMBEDDINGS_CACHE = "data/embeddings.npy"  # опционально
# -------------------------

def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-нормализация строк (each row = vector)."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    # предотвращаем деление на 0
    norms[norms == 0] = 1.0
    return emb / norms

def build_model(model_name: str = MODEL_NAME, device: str = "cpu"):
    """Инициализация модели sentence-transformers."""
    model = SentenceTransformer(model_name, device=device)
    return model

def encode_chunks(model: SentenceTransformer, chunks: list) -> np.ndarray:
    """
    chunks: list of dicts, каждый: {"chunk_id": str_or_int, "clean_text": str, "raw_text": str, "metadata": {...}}
    Возвращает np.array shape (N, D)
    """
    texts = [c["clean_text"] for c in chunks]
    # без батчей — encode весь список
    embeddings = model.encode(["passage: " + t for t in texts], show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_and_save_faiss_index(chunks: list,
                               model_name: str = MODEL_NAME,
                               index_path: str = INDEX_PATH,
                               metadata_path: str = METADATA_PATH,
                               embeddings_cache: str = EMBEDDINGS_CACHE,
                               use_cache: bool = True):
    # 1) модель
    model = build_model(model_name)

    # 2) получить эмбеддинги (или загрузить из кэша)
    if use_cache and os.path.exists(embeddings_cache):
        print("USED CAHCE")
        emb = np.load(embeddings_cache)
    else:
        emb = encode_chunks(model, chunks)
        np.save(embeddings_cache, emb)

    # 3) нормализация для cosine
    emb = normalize_embeddings(emb).astype('float32')

    # 4) создаём FAISS index для inner product (cosine on normalized vectors)
    d = emb.shape[1]
    index_flat = faiss.IndexFlatIP(d)  # exact search, для 500 векторов — идеально
    # если хочется хранить id → используем IndexIDMap
    index = faiss.IndexIDMap(index_flat)

    # 5) подготовка id (числовые)
    ids = np.array([int(i) for i, _ in enumerate(chunks)], dtype='int64')

    # 6) add
    index.add_with_ids(emb, ids)

    # 7) сохранить индекс
    faiss.write_index(index, index_path)

    # 8) сохранить метаданные: мап id -> chunk
    # id2meta = {}
    # for i, chunk in enumerate(chunks):
    #     id2meta[str(i)] = {
    #         "chunk_id": chunk.get("chunk_id"),
    #         "raw_text": chunk.get("raw_text"),
    #         "metadata": chunk.get("metadata")
    #     }
    # with open(metadata_path, "w", encoding="utf-8") as f:
    #     json.dump(id2meta, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index to {index_path}")

def load_index_and_metadata(index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
    index = faiss.read_index(index_path)
    # with open(metadata_path, "r", encoding="utf-8") as f:
    #     id2meta = json.load(f)
    return index#, id2meta

def search(index, chunks, query: str, model_name: str = MODEL_NAME, top_k: int = 5):
    model = build_model(model_name)
    q_emb = model.encode(["query: " + query], convert_to_numpy=True)
    q_emb = normalize_embeddings(q_emb.astype('float32'))

    D, I = index.search(q_emb, top_k)  # D = scores (inner product), I = ids
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({
            "faiss_id": int(idx),
            "score": float(score),   # since IP on normalized -> cosine in [-1,1]
            "chunk_id": chunks[idx].get("chunk_id"),
            "raw_text": chunks[idx].get("raw_text"),
            "metadata": chunks[idx].get("metadata")
        })
    return results

def load_chunks(fname = "chunks.jsonl"):
    data = []

    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
    return data
# -------------------------
# Пример использования (скелет):
# chunks = load_your_chunks_somehow()  # список словарей, по формату описанному выше
# build_and_save_faiss_index(chunks)
# index, meta = load_index_and_metadata()
# print(search(index, meta, "Какова формула?", top_k=5))
# -------------------------

if __name__ == "__main__":
    chunks = load_chunks("chunks.jsonl")
    build_and_save_faiss_index(chunks)
    index = load_index_and_metadata()
    res = search(index, chunks, "Из чего состоит мультимножество", top_k=3)
    for ch in res:
        print(ch)
        print()
