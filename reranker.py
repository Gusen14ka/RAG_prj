from sentence_transformers import CrossEncoder
from typing import Optional
from entities import Chunk

MODEL_NAME = "models/bge-reranker-v2-m3"


def build_reranker(model_name: str = MODEL_NAME) -> CrossEncoder:
    """Загружает CrossEncoder один раз."""
    return CrossEncoder(model_name)


def rerank(
        query: str,
        chunks: list[Chunk],
        top_k: int,
        model: Optional[CrossEncoder] = None,  # ИСПРАВЛЕНО: принимаем готовую модель
) -> list[tuple[str, float]]:              # ИСПРАВЛЕНО: правильная аннотация типа

    if model is None:
        model = build_reranker()

    pairs = []
    chunks_ids = []
    for ch in chunks:
        text = ch.raw_text
        pairs.append([query, text])
        chunks_ids.append(ch.chunk_id)

    if not pairs:
        return []

    scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

    return sorted(zip(chunks_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]