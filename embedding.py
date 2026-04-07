# embed_and_index.py
#
# faiss заменён на numpy dot-product — faiss конфликтует с CrossEncoder (OpenMP)
# на macOS ARM: faiss.read_index блокирует OMP thread pool, после чего
# CrossEncoder не может инициализировать свой. Для ~500 чанков numpy быстрее.

from sentence_transformers import SentenceTransformer
import numpy as np
import json

# -------------------------
MODEL_NAME       = "models/multilingual-e5-large"
EMBEDDINGS_PATH  = "data/embeddings.npy"   # хранилище векторов
# -------------------------


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-нормализация строк."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def build_model(model_name: str = MODEL_NAME, device: str = "cpu") -> SentenceTransformer:
    """Загружает embedding-модель."""
    return SentenceTransformer(model_name, device=device)


def encode_chunks(model: SentenceTransformer, chunks: list) -> np.ndarray:
    """Кодирует чанки в матрицу эмбеддингов (N, D)."""
    texts = [c["clean_text"] for c in chunks]
    # show_progress_bar=False — иначе потоки tqdm/torch конфликтуют с CrossEncoder
    return model.encode(["passage: " + t for t in texts],
                        show_progress_bar=False,
                        convert_to_numpy=True)


def build_and_save_embeddings(chunks: list,
                               model: SentenceTransformer,
                               embeddings_path: str = EMBEDDINGS_PATH) -> np.ndarray:
    """Строит нормализованные эмбеддинги и сохраняет в .npy файл."""
    emb = encode_chunks(model, chunks)
    emb = normalize_embeddings(emb).astype("float32")
    np.save(embeddings_path, emb)
    print(f"Сохранено эмбеддингов: {emb.shape} → {embeddings_path}")
    return emb


def load_embeddings(embeddings_path: str = EMBEDDINGS_PATH) -> np.ndarray:
    """Загружает матрицу эмбеддингов из .npy файла."""
    return np.load(embeddings_path)


def search(embeddings: np.ndarray, chunks: list, query: str,
           model: SentenceTransformer, top_k: int = 5) -> list:
    """
    Косинусный поиск через numpy dot product (normalized vectors).
    embeddings: матрица (N, D) с L2-нормализованными векторами чанков.
    """
    q_emb = model.encode(["query: " + query], convert_to_numpy=True).astype("float32")
    q_emb = normalize_embeddings(q_emb)           # shape (1, D)

    scores = (embeddings @ q_emb.T).ravel()       # cosine similarity (N,)
    top_ids = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_ids:
        results.append({
            "faiss_id": int(idx),
            "score":    float(scores[idx]),
            "chunk_id": chunks[idx].get("chunk_id"),
            "raw_text": chunks[idx].get("raw_text"),
            "metadata": chunks[idx].get("metadata"),
        })
    return results


def load_chunks(fname: str = "chunks.jsonl") -> list:
    data = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    chunks = load_chunks("data/chunks.jsonl")
    model  = build_model()
    emb    = build_and_save_embeddings(chunks, model)
    res    = search(emb, chunks, "Из чего состоит мультимножество", model, top_k=3)
    for ch in res:
        print(ch)
        print()