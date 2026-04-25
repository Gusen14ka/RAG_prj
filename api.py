from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from collections import OrderedDict
import asyncio
import threading
import time
import uvicorn

from pipeline import (
    build_search_pipeline,
    QUESTIONS_BANK_PATH,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    RERANK_THRESHOLD,
    ANSWER_THRESHOLD,
    Q_SCORE_WEIGHT,
    SRC_SCORE_WEIGHT,
    GOLD_SCORE_WEIGHT,
)
from embedding import search as embedding_search
from keyword_search import search_bm25
from utils.rrf_scoring import rrf_scoring
from utils.load_chunks import load_chunks
from reranker import rerank
from trainer_mode import Trainer, Question


@asynccontextmanager
async def lifecycle(app: FastAPI):
    # build_search_pipeline запускается синхронно при старте — это нормально,
    # так как lifespan выполняется до приёма запросов.
    state = build_search_pipeline(verbose=True)

    trainer = Trainer(
        state["rerank_model"], str(QUESTIONS_BANK_PATH),
        ANSWER_THRESHOLD, Q_SCORE_WEIGHT, SRC_SCORE_WEIGHT, GOLD_SCORE_WEIGHT,
    )
    questions: list[Question] = []

    if not QUESTIONS_BANK_PATH.exists():
        print("    Генерируем вопросы...")
        for chunk_id, chunk in state["chunks_with_key"].items():
            generated = trainer.build_question_bank(chunk["clean_text"], chunk_id, min_score=0.65)
            questions.extend(generated)
        print(f"    Сгенерировано {len(questions)} вопросов.")
        trainer.save_questions(questions)
    else:
        print("    Загружаем вопросы из файла...")
        questions_dicts = load_chunks(str(QUESTIONS_BANK_PATH))
        questions = [Question(**d) for d in questions_dicts]
        print(f"    Загружено {len(questions)} вопросов.")

    app.state.chunks          = state["chunks"]
    app.state.chunks_with_key = state["chunks_with_key"]
    app.state.embed_model     = state["embed_model"]
    app.state.rerank_model    = state["rerank_model"]
    app.state.embeddings      = state["embeddings"]
    app.state.bm25            = state["bm25"]
    app.state.trainer         = trainer
    app.state.questions       = questions

    print("    Lifespan complete.")
    yield


def perform_search(embed_model, rerank_model, bm25, chunks, chunks_with_key, embeddings, query):
    cached = _cache_get(query)
    if cached is not None:
        print(f"[SEARCH] cache hit: '{query[:60]}'")
        return cached

    t0 = time.perf_counter()
    res_embedding = embedding_search(embeddings, chunks, query,
                                     model=embed_model, top_k=TOP_K_RETRIEVAL)
    t_embed = time.perf_counter()

    res_keyword = search_bm25(bm25, chunks, query, top_k=TOP_K_RETRIEVAL)
    t_bm25 = time.perf_counter()

    res_rrf_ids = rrf_scoring([res_embedding, res_keyword], weight_list=[1, 1])
    res_rrf     = [chunks_with_key[x[0]] for x in res_rrf_ids]
    t_rrf = time.perf_counter()

    res_rerank_ids = rerank(query, res_rrf, TOP_K_RERANK, model=rerank_model)
    t_rerank = time.perf_counter()

    filtered = [(cid, score) for cid, score in res_rerank_ids if score >= RERANK_THRESHOLD]

    embed_ms  = (t_embed  - t0)      * 1000
    bm25_ms   = (t_bm25   - t_embed) * 1000
    rrf_ms    = (t_rrf    - t_bm25)  * 1000
    rerank_ms = (t_rerank - t_rrf)   * 1000
    total_ms  = (t_rerank - t0)      * 1000
    print(
        f"[SEARCH] embed={embed_ms:.0f}ms  bm25={bm25_ms:.0f}ms  "
        f"rrf={rrf_ms:.0f}ms  rerank={rerank_ms:.0f}ms  total={total_ms:.0f}ms"
    )

    if not filtered:
        return []

    result = [
        {"text": chunks_with_key[cid].get("raw_text", "").strip(),
         "chunk_id": cid,
         "score": float(score)}
        for cid, score in filtered
    ]
    _cache_set(query, result)
    return result


def perform_check_answer(question_id, trainer, questions, user_answer):
    question = next((q for q in questions if q.question_id == question_id), None)
    if not question:
        return {"error": "Question not found"}

    is_correct, score = trainer.check_answer(user_answer, question)
    return {
        "is_correct":      bool(is_correct),
        "score":           float(score),
        "source_sentence": question.source_sentence,
        "gold_answer":     question.answer,
    }


# ---------------------------------------------------------------------------
# LRU query cache
# ---------------------------------------------------------------------------
_CACHE_MAX_SIZE = 64
_search_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: str):
    with _cache_lock:
        if key not in _search_cache:
            return None
        _search_cache.move_to_end(key)
        return _search_cache[key]


def _cache_set(key: str, value) -> None:
    with _cache_lock:
        if key in _search_cache:
            _search_cache.move_to_end(key)
        _search_cache[key] = value
        if len(_search_cache) > _CACHE_MAX_SIZE:
            _search_cache.popitem(last=False)


app = FastAPI(title="NON LLM RAG", version="1.0.0", lifespan=lifecycle)

# CORS — нужен при запуске фронтенда через Live Server (Go Live)
# В Docker запросы идут через nginx в той же origin — CORS не нужен
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class CheckAnswerRequest(BaseModel):
    question_id: str
    user_answer: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search")
async def search(body: SearchRequest, request: Request):
    try:
        result = await asyncio.to_thread(
            perform_search,
            request.app.state.embed_model,
            request.app.state.rerank_model,
            request.app.state.bm25,
            request.app.state.chunks,
            request.app.state.chunks_with_key,
            request.app.state.embeddings,
            body.query,
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check_answer")
async def check_answer(body: CheckAnswerRequest, request: Request):
    try:
        result = await asyncio.to_thread(
            perform_check_answer,
            body.question_id,
            request.app.state.trainer,
            request.app.state.questions,
            body.user_answer,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/questions")
async def get_question(request: Request):
    def serialize_question(q):
        d = q.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, (int, float)) and hasattr(v, 'item'):  # numpy types
                d[k] = float(v)
        return d
    return {"questions": [serialize_question(q) for q in request.app.state.questions]}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
