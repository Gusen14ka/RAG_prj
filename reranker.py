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
        model: Optional[CrossEncoder] = None,
) -> list[tuple[Chunk, float]]:

    if model is None:
        model = build_reranker()

    pairs = []
    for ch in chunks:
        text = ch.raw_text
        pairs.append([query, text])

    if not pairs:
        return []

    scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

    return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]