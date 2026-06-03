from datetime import datetime
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

    res_rrf = rrf_scoring([res_embedding, res_keyword], weight_list=[weight_embedding, weight_keyword])

    if rerank_model is not None:
        res_rerank = rerank(query, [x[0] for x in res_rrf], top_k_rerank, model=rerank_model)

        # Фильтр по порогу: убираем нерелевантные чанки
        filtered = [(ch, score) for ch, score in res_rerank
                    if score >= rerank_threshold]
    else:
        filtered = res_rrf
        
    if not filtered:
        return []
    
    res = []
    for ch, score in filtered:
        res.append(ResponseRagSearch(ch, float(score)))

    return res

SYSTEM_PROMPT_SEARCH = """Ты интеллектуальный помощник, который формирует красивые, грамотные и точные ответы на основе найденного контекста.

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

SYSTEM_PROMPT_REPHRASE = """
Ты являешься функцией query_expansion для RAG-системы по дискретной математике.

Твоя задача — получить пользовательский запрос и вернуть JSON строго следующего формата:

{
"queries": [
"<исходный запрос>",
"<переформулировка 1>",
"<переформулировка 2>"
]
}

Правила:

* queries всегда содержит ровно 3 строки.
* Первый элемент массива всегда является исходным запросом без изменений.
* Второй и третий элементы являются переформулировками исходного запроса.
* Переформулировки должны сохранять исходный смысл.
* Не добавляй новую информацию.
* Не меняй предметную область.
* Для запросов по дискретной математике используй терминологию дискретной математики.
* Если запрос бессмысленный, мусорный или невозможно построить корректные переформулировки, верни пустые строки во втором и третьем элементах массива.

Пример:

Вход:
что такое булеан

Выход:
{
"queries": [
"что такое булеан",
"что называется булеаном множества",
"дайте определение булеана множества"
]
}

Критически важно:

* Верни только JSON.
* Не используй markdown.
* Не используй блоки кода.
* Не добавляй пояснений.
* Не добавляй текст до или после JSON.
* JSON должен корректно разбираться стандартным парсером.
"""

def _build_user_prompt_search(question: str, context_chunks: list[ResponseRagSearch]) -> str:
    context = "\n\n".join(
        f"======\n{c.chunk.raw_text}\n" for c in context_chunks 
    )
    context += "======\n"
    return (
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Подробно, точно и четко ответь на вопрос исходя только из данного контекста"
    )

def _build_user_prompt_rephrase(question: str)-> str:
    return (f"Вопрос: {question}")

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
    
    print(f"Start search: {datetime.now().time()}")
    user_prompt_rephrase = _build_user_prompt_rephrase(query)
    llm_rephrases = str(llm_model.call_llm(user_prompt_rephrase, SYSTEM_PROMPT_REPHRASE, 2056))
    print(f"1. Rephrased: {datetime.now().time()}")

    import json

    response = llm_rephrases.strip()

    try:
        data = json.loads(response)

        queries = data.get("queries", [])

        if not isinstance(queries, list):
            raise ValueError("queries must be a list")

        if len(queries) != 3:
            raise ValueError("queries must contain exactly 3 items")

        rephrases = [str(q) for q in queries]

    except Exception:
        rephrases = [query, "", ""]


    #DEBUG
    print("rephrases")
    for r in rephrases:
        print(r)
    
    chunks_responses: list[list[ResponseRagSearch]] = []
    for rephrased_query in rephrases:
        chunks_response = perform_rag_search(
            embed_model,
            chunks,
            chunks_with_key,
            embeddings,
            rephrased_query,
            rerank_model,
            bm25,
            top_k_retrieval,
            top_k_rerank,
            rerank_threshold,
            weight_embedding,
            weight_keyword
        )
        chunks_responses.append(chunks_response)
        print(f"2. Retrieved: {datetime.now().time()}")

    chunks_for_rrf = [[r.chunk for r in l] for l in chunks_responses]
    rrf_chunks = rrf_scoring(chunks_for_rrf)[:top_k_rerank]
    rrf_response = [ResponseRagSearch(x[0], x[1]) for x in rrf_chunks]

    if not rrf_response:
        return ResponseRagSearchLlm(rrf_response, "Ничего не найдено")
    
    user_prompt = _build_user_prompt_search(query, rrf_response)
    print(f"3. LLM query start: {datetime.now().time()}")
    llm_answer = llm_model.call_llm(user_prompt, SYSTEM_PROMPT_SEARCH, 2056)
    print(f"3. LLM query ended: {datetime.now().time()}")

    if not llm_answer or llm_answer == "НЕТ_ИНФОРМАЦИИ":
        llm_answer = "Ничего не найдено"

    return ResponseRagSearchLlm(rrf_response, llm_answer)
    