from concurrent.futures import Future
from dataclasses import asdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from keyword_search import build_bm25
from parsing_terms import extract_terms
from config.constants import TOP_K_RETRIEVAL, TOP_K_RERANK, RERANK_THRESHOLD
from search_pipeline import perform_rag_search
from entities import (
    ResponseRagSearch,
    Chunk,
    Question
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты интеллектуальный помощник, который формирует красивые, грамотные и краткие ответы на основе найденного контекста.

    Ты отвечаешь строго по контексту.

    Сначала извлеки смысл ответа из контекста.
    Затем сформулируй ровно 3 варианта ответа на русском языке.

    Правила:
    - использовать только контекст, данный пользователем;
    - не добавлять знания от себя;
    - не использовать общие энциклопедические формулировки, если они не следуют из текста;
    - каждый вариант — отдельная строка;
    - каждый вариант — полное предложение;
    - каждый ответ — четкий ответ на вопрос пользователя, без воды 
    - варианты могут быть близки по смыслу, но должны быть сформулированы по-разному;
    - нельзя возвращать меньше 3 строк;
    - нельзя объединять варианты в одну строку;
    - если построить ответ только по данному пользователем контексту невозможно, вернуть 3 строки равные: НЕТ_ИНФОРМАЦИИ

    Если не уверен можно ли построить ответ или нет, лучше не строй, а возвращай 3 строки равные: НЕТ_ИНФОРМАЦИИ
    """


def _section_id_of_chunk(chunk: Chunk) -> str:
    """Возвращает section_id чанка из его chunk_id ('1.4.1:3' → '1.4.1')."""
    return chunk.chunk_id.split(":")[0]

def _create_bm25_indices_for_sections(
        chunks: list[Chunk],
        section_ids: list[str]
) -> dict[str, Optional[BM25Okapi]]:
    
    bm25_indices: dict[str, Optional[BM25Okapi]] = {}
    for s_id in section_ids:
        indices = [i for i, c in enumerate(chunks) if _section_id_of_chunk(c) == s_id]

        filtered_chunks = [chunks[i] for i in indices]
        if len(filtered_chunks) == 0:
            bm25_indices[s_id] = None
        else:
            bm25 = build_bm25(filtered_chunks)
            bm25_indices[s_id] = bm25

    return bm25_indices

def _filter_by_section(
    chunks: list[Chunk],
    chunks_with_key: dict[str, Chunk],
    embeddings: np.ndarray,
    section_id: str,
) -> tuple[list[Chunk], dict[str, Chunk], np.ndarray]:
    """
    Оставляет только чанки нужного раздела.

    Возвращает (filtered_chunks, filtered_chunks_with_key, filtered_embeddings).

    chunks и embeddings связаны по индексу: embeddings[i] ↔ chunks[i].
    chunks_with_key — словарь {chunk_id: <данные>}, фильтруется по prefix chunk_id.
    """
    indices = [i for i, c in enumerate(chunks) if _section_id_of_chunk(c) == section_id]

    filtered_chunks = [chunks[i] for i in indices]
    filtered_embeddings = (
        embeddings[indices] if indices else np.empty((0, embeddings.shape[1]))
    )
    filtered_chunks_with_key = {
        k: v for k, v in chunks_with_key.items()
        if k.split(":")[0] == section_id
    }

    return filtered_chunks, filtered_chunks_with_key, filtered_embeddings


def _build_user_prompt(question: str, context_chunks: list[ResponseRagSearch]) -> str:
    context = "\n\n".join(
        f"[Чанк {i + 1}]\n{c.chunk.raw_text}"
        for i, c in enumerate(context_chunks)
    )
    return (
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Ответ должен быть извлечён только из контекста. "
        "Если ответ нельзя уверенно сформулировать по контексту, верни НЕТ_ИНФОРМАЦИИ."
    )

def _search_for_term(
    term: str,
    section_id: str,
    chunks: list[Chunk],
    chunks_with_key: dict[str, Chunk],
    embeddings: np.ndarray,
    embed_model,
    bm25: Optional[BM25Okapi],
    rerank_model,
) -> tuple[str, list[ResponseRagSearch]]:
    """
    Ищет релевантные чанки для одного термина в его разделе.
    Возвращает (question, found_chunks).
    Запускается в пуле потоков.
    """
    question = f"Что такое {term}?"

    filtered_chunks, filtered_chunks_with_key, filtered_embeddings = (
        _filter_by_section(chunks, chunks_with_key, embeddings, section_id)
    )

    if not filtered_chunks:
        logger.warning("Нет чанков для раздела %s (термин: %s)", section_id, term)
        return question, []

    found_chunks = perform_rag_search(
        embed_model=embed_model,
        chunks=filtered_chunks,
        chunks_with_key=filtered_chunks_with_key,
        embeddings=filtered_embeddings,
        query=question,
        rerank_model=rerank_model,
        bm25=bm25,
        top_k_retrieval=TOP_K_RETRIEVAL,
        top_k_rerank=TOP_K_RERANK,
        rerank_threshold=RERANK_THRESHOLD,
    )

    return question, found_chunks


def generate_questions(
        pdf_path: str,
        embedding_model,
        rerank_model,
        chunks: list[Chunk],
        chunks_with_key: dict[str, Chunk],
        embeddings: np.ndarray,
        llm_model,
        questions_bank_path: str,
        max_workers: int = 4,
) -> None:
    """
    Для каждого термина из PDF:
      1. Формирует вопрос «Что такое X?»
      2. Параллельно ищет релевантные чанки только из раздела термина
      3. Генерирует 3 варианта ответа через LLM (последовательно, GPU-bound)
      4. Сохраняет список {term, section_id, question, answers, chunk_ids} в JSON

    Параметры
    ----------
    pdf_path            : путь к PDF учебника
    embedding_model     : модель эмбеддингов
    rerank_model        : модель реранкинга
    chunks              : list[Chunk] — все чанки; chunks[i] ↔ embeddings[i]
    chunks_with_key     : dict[chunk_id, ...] — чанки по chunk_id для поиска
    embeddings          : np.ndarray shape (N, D), загруженный из embedding.npy
    llm_model           : объект с методом call_llm(query, system_promt, max_new_tokens)
    questions_bank_path : путь для сохранения результирующего JSON
    max_workers         : число потоков для параллельного поиска
    """

    # ── 1. Термины ────────────────────────────────────────────────────────────
    terms = extract_terms(pdf_path)   # list[tuple[str, str]] → (term, section_id)
    #terms = [("изоморфизм упорядоченных множеств", "1.8.8"), ("Теорема о наименьшей неподвижной точке функции", "1.8.7")]
    logger.info("Терминов найдено: %d", len(terms))
    print(f"Термины созданы: {len(terms)}")

    # ── 2. Параллельный поиск чанков ─────────────────────────────────────────
    print("Ищём подходящие чанки...")

    # Используем (term, section_id) как ключ на случай одинаковых терминов
    # в разных разделах
    futures: dict[Future[tuple[str, list[ResponseRagSearch]]], tuple[str, str]] = {}
    results_raw: dict[tuple[str, str], tuple[str, list[ResponseRagSearch]]] = {}

    # Подготовим индексы bm25
    section_ids = [t[1] for t in terms]
    bm25_indices = _create_bm25_indices_for_sections(chunks, section_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for term, section_id in terms:
            future = pool.submit(
                _search_for_term,
                term, section_id,
                chunks, chunks_with_key, embeddings,
                embedding_model, bm25_indices[section_id], rerank_model,
            )
            futures[future] = (term, section_id)

        for future in as_completed(futures):
            term, section_id = futures[future]
            try:
                question, found_chunks = future.result()
                results_raw[(term, section_id)] = (question, found_chunks)
            except Exception as exc:
                logger.error("Ошибка поиска для '%s' [%s]: %s", term, section_id, exc)
                results_raw[(term, section_id)] = (f"Что такое {term}?", [])

    # ── 3. Генерация ответов через LLM ───────────────────────────────────────
    print("Генерируем ответы через LLM...")
    questions_bank: list[Question]= []

    term_id = 0
    for term, section_id in terms:      # исходный порядок терминов
        term_id += 1
        question, found_chunks = results_raw.get(
            (term, section_id), (f"Что такое {term}?", [])
        )

        if not found_chunks:
            logger.warning("Пропускаем '%s' [%s] — нет чанков", term, section_id)
            continue

        user_prompt = _build_user_prompt(question, found_chunks)

        try:
            raw_answer: str = llm_model.call_llm(
                query=user_prompt,
                system_promt=SYSTEM_PROMPT,
                max_new_tokens=512,
            )
        except Exception as exc:
            logger.error("LLM ошибка для '%s': %s", term, exc)
            raw_answer = ""

        if raw_answer == "НЕТ_ИНФОРМАЦИИ":
            raw_answer = ""
        answers = [line.strip() for line in raw_answer.splitlines() if line.strip()]
        chunk_ids = [c.chunk.chunk_id for c in found_chunks]

        questions_bank.append(Question(term_id, term, question, answers, chunk_ids))

        logger.debug("[%s] %s → %d вариантов", section_id, term, len(answers))

    # ── 4. Сохраняем ─────────────────────────────────────────────────────────
    with open(questions_bank_path, "w", encoding="utf-8") as f:
        data = []
        for q in questions_bank:
            if "НЕТ_ИНФОРМАЦИИ" not in q.answers and "" not in q.answers:
                data.append(asdict(q))
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Банк вопросов сохранён: {questions_bank_path} ({len(questions_bank)} записей)")