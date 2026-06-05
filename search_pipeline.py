from datetime import datetime
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from time import perf_counter

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
    t0 = perf_counter()
    res_embedding = embedding_search(embeddings, chunks, query,
                                         model=embed_model, top_k=top_k_retrieval)
    print(f"embadding search = {perf_counter() - t0}")

    t0 = perf_counter()
    if bm25 is not None:
        res_keyword = search_bm25(bm25, chunks, query, top_k=top_k_retrieval)
    else:
        res_keyword = []
    print(f"bm25 search = {perf_counter() - t0}")

    res_rrf = rrf_scoring([res_embedding, res_keyword], weight_list=[weight_embedding, weight_keyword])

    t0 = perf_counter()
    if rerank_model is not None:
        res_rerank = rerank(query, [x[0] for x in res_rrf], top_k_rerank, model=rerank_model)

        # Фильтр по порогу: убираем нерелевантные чанки
        filtered = [(ch, score) for ch, score in res_rerank
                    if score >= rerank_threshold]
    else:
        filtered = res_rrf

    print(f"rerank = {perf_counter() - t0}")
        
    if not filtered:
        return []
    
    res = []
    for ch, score in filtered:
        res.append(ResponseRagSearch(ch, float(score)))

    return res

SYSTEM_PROMPT_SEARCH = """Ты интеллектуальный помощник, который извлекает грамотные и точные ответы из предоставленного контекста.

    Ты отвечаешь только по данному контексту.

    Правила:
    - используй только информацию, которая прямо есть в контексте;
    - не добавляй примеры, пояснения, оговорки и дополнительные факты;
    - не перефразируй определение слишком свободно: сохраняй смысл и терминологию контекста;
    - если в контексте нет прямого ответа на вопрос, верни ровно: НЕТ_ИНФОРМАЦИИ;
    - если ответ можно дать, дай его кратко, в 1–2 предложениях;
    - не используй общие энциклопедические формулировки, если их нет в контексте;
    - не выдумывай обозначения, формулы и свойства.

    Формат ответа:
    - либо краткий ответ по контексту;
    - либо ровно: НЕТ_ИНФОРМАЦИИ
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
        weight_keyword = 1,
        use_rephrase: bool = True,
) -> ResponseRagSearchLlm:
    
    print(f"Start search: {datetime.now().time()}")

    if use_rephrase:
        user_prompt_rephrase = _build_user_prompt_rephrase(query)
        llm_rephrases = str(llm_model.call_llm(user_prompt_rephrase, SYSTEM_PROMPT_REPHRASE, 100))
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
    else:
        rephrases = [query, "", ""]


    # Убираем пустые перефразировки — они ведут к лишним запросам и
    # могут вызывать проблемы при параллельном кодировании запросов.
    rephrases = [r for r in rephrases if r and r.strip()]

    #DEBUG
    print("rephrases")
    for r in rephrases:
        print(r)

    def _retrieve(rephrased_query: str) -> list[ResponseRagSearch]:
        return perform_rag_search(
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
    chunks_responses: list[list[ResponseRagSearch]] = [None] * len(rephrases)  # type: ignore

    # Если только один запрос, выполняем последовательно — это надежнее
    # для SentenceTransformer, который не всегда корректно работает в нескольких потоках.
    if len(rephrases) == 1:
        chunks_responses[0] = _retrieve(rephrases[0])
        print(f"2. Retrieved [0]: {datetime.now().time()}, chunks:{len(chunks_responses[0])}")
    else:
        with ThreadPoolExecutor(max_workers=len(rephrases)) as executor:
            future_to_idx = {executor.submit(_retrieve, q): i for i, q in enumerate(rephrases)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunks_responses[idx] = future.result()
                print(f"2. Retrieved [{idx}]: {datetime.now().time()}, chunks:{len(chunks_responses[idx])}")

    chunks_for_rrf = [[r.chunk for r in l] for l in chunks_responses]
    rrf_chunks = rrf_scoring(chunks_for_rrf)[:top_k_rerank]
    rrf_response = [ResponseRagSearch(x[0], x[1]) for x in rrf_chunks]

    if not rrf_response:
        return ResponseRagSearchLlm(rrf_response, "Ничего не найдено")
    
    user_prompt = _build_user_prompt_search(query, rrf_response)
    print(f"3. LLM query start: {datetime.now().time()}")
    llm_answer = llm_model.call_llm(user_prompt, SYSTEM_PROMPT_SEARCH, 150)
    print(f"3. LLM query ended: {datetime.now().time()}")

    if not llm_answer or llm_answer == "НЕТ_ИНФОРМАЦИИ":
        llm_answer = "Ничего не найдено"

    return ResponseRagSearchLlm(rrf_response, llm_answer)
    