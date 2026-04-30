from typing import Any

from embedding import search as embedding_search
from keyword_search import search_bm25
from utils.rrf_scoring import rrf_scoring
from reranker import rerank
from entities import (
    ResponseRagSearch,
    ResponseRagSearchLlm,
    Chunk
)

def perform_rag_search(
        embed_model,
        chunks: list[Chunk],
        chunks_with_key: dict[str, Chunk],
        embeddings,
        query: str,
        rerank_model : Any = None,
        bm25 : Any = None,
        top_k_retrieval = 5,
        top_k_rerank = 3,
        rerank_threshold = 0.3,
        weight_embedding = 1,
        weight_keyword = 1
) -> list[ResponseRagSearch]:
    res_embedding = embedding_search(embeddings, chunks, query,
                                         model=embed_model, top_k=top_k_retrieval)
    if bm25 is not None:
        res_keyword = search_bm25(bm25, chunks, query, top_k=top_k_retrieval)
    else:
        res_keyword = []

    res_rrf_ids = rrf_scoring([res_embedding, res_keyword], weight_list=[weight_embedding, weight_keyword])
    res_rrf = [chunks_with_key[x[0]] for x in res_rrf_ids]

    if rerank_model is not None:
        res_rerank_ids = rerank(query, res_rrf, top_k_rerank, model=rerank_model)

        # Фильтр по порогу: убираем нерелевантные чанки
        filtered = [(cid, score) for cid, score in res_rerank_ids
                    if score >= rerank_threshold]
    else:
        filtered = res_rrf_ids
        
    if not filtered:
        return []
    
    res = []
    for cid, score in filtered:
        ch   = chunks_with_key[cid]
        res.append(ResponseRagSearch(ch, float(score)))

    return res

SYSTEM_PROMPT = """Ты интеллектуальный помощник, который формирует красивые, грамотные и точные ответы на основе найденного контекста.

    Ты отвечаешь строго по контексту.

    Сначала извлеки смысл ответа из контекста.
    Затем сформулируй максимально понятный ответ на русском языке.

    Правила:
    - использовать только контекст, данный пользователем;
    - не добавлять знания от себя;
    - не использовать общие энциклопедические формулировки, если они не следуют из текста;
    - если построить ответ только по данному пользователем контексту невозможно, вернуть ровно: НЕТ_ИНФОРМАЦИИ

    Если не уверен можно ли построить ответ или нет, лучше не строй, а возвращай ровно: НЕТ_ИНФОРМАЦИИ
    """

def _build_user_prompt(question: str, context_chunks: list[ResponseRagSearch]) -> str:
    context = "\n\n".join(
        f"======\n{c.chunk.raw_text}\n" for c in context_chunks 
    )
    context += "======\n"
    return (
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Подробно, точно и четко ответь на вопрос исходя только из данного контекста"
    )

def perform_rag_search_llm(
        llm_model,
        embed_model,
        chunks: list[Chunk],
        chunks_with_key: dict[str, Chunk],
        embeddings,
        query: str,
        rerank_model : Any = None,
        bm25 : Any = None, 
        top_k_retrieval = 5,
        top_k_rerank = 3,
        rerank_threshold = 0.3,
        weight_embedding = 1,
        weight_keyword = 1        
) -> ResponseRagSearchLlm:
    chunks_response = perform_rag_search(
        embed_model,
        chunks,
        chunks_with_key,
        embeddings,
        query,
        rerank_model,
        bm25,
        top_k_retrieval,
        top_k_rerank,
        rerank_threshold,
        weight_embedding,
        weight_keyword
    )
    if not chunks_response:
        return ResponseRagSearchLlm(chunks_response, "Ничего не найдено")
    
    user_prompt = _build_user_prompt(query, chunks_response)
    llm_answer = llm_model.call_llm(user_prompt, SYSTEM_PROMPT, 2056)

    if not llm_answer or llm_answer == "НЕТ_ИНФОРМАЦИИ":
        llm_answer = "Ничего не найдено"

    return ResponseRagSearchLlm(chunks_response, llm_answer)
    