from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
MODEL_NAME = "models/bge-reranker-v2-m3"

def rerank(
        query: str, 
        chunks: List[Dict[str, Any]],
        top_k: int) -> List[str]:
    
    reranker = CrossEncoder(MODEL_NAME)
    
    pairs = []
    chunks_ids = []
    for ch in chunks:
        text = ch["raw_text"]
        pairs.append([query, text])
        chunks_ids.append(ch["chunk_id"])

    if not pairs:
        return []
    
    scores = reranker.predict(pairs, show_progress_bar=True)

    return sorted(zip(chunks_ids, scores), key=lambda x: x[0], reverse=True)[:top_k]
