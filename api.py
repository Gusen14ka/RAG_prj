from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from collections import OrderedDict
import asyncio
import threading
from pathlib import Path
import os
import uvicorn

from initialization import init_rag
from check_answer import perform_check_answer
from search_pipeline import perform_rag_search, perform_rag_search_llm
from config.paths import (
    CHUNK_PATH,
    CHUNK_WITH_KEY_PATH,
    SOURCE_PDF_PATH,
    EMBEDDINGS_PATH,
    BM25_PATH,
    QUESTIONS_BANK_PATH,
    EMBADDING_MODEL_PATH,
    RERANK_MODEL_PATH,
    LLM_MODEL_PATH
)
from api_schemas import Question
from entities import Question as DtoQusetion
from converters import (
    convert_question,
    convert_check_answer,
    convert_rag_search,
    convert_rag_search_llm
)

@asynccontextmanager
async def lifecycle(app: FastAPI):
    state = init_rag()

    # Сохраняем все необходимые объекты
    app.state.chunks = state.chunks
    app.state.chunks_with_key = state.chunks_with_key
    app.state.embed_model = state.embed_model
    app.state.rerank_model = state.rerank_model
    app.state.embeddings = state.embeddings
    app.state.bm25 = state.bm25
    app.state.questions = [convert_question(q) for q in state.question_dtos]
    app.state.question_dtos = state.question_dtos
    app.state.llm_model = state.llm_model

    print("    Lifespan complete.")

    yield


_CACHE_MAX_SIZE = 64
_search_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: str):
    with _cache_lock:
        if key not in _search_cache:
            return None
        _search_cache.move_to_end(key)
        return _search_cache[key]


def _cache_add(key: str, value) -> None:
    with _cache_lock:
        if key in _search_cache:
            _search_cache.move_to_end(key)
        _search_cache[key] = value
        if len(_search_cache) > _CACHE_MAX_SIZE:
            _search_cache.popitem(last=False)

app = FastAPI(title="RAG", version="1.0.0", lifespan=lifecycle)

class SearchRequest(BaseModel):
    query: str

class CheckAnswerRequest(BaseModel):
    question_id: int
    user_answer: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/search")
async def search(body: SearchRequest, request: Request):
    cached = _cache_get(f"NONLLM{body.query}")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(
            perform_rag_search,
            request.app.state.embed_model,
            request.app.state.chunks,
            request.app.state.chunks_with_key,
            request.app.state.embeddings,
            body.query,
            request.app.state.rerank_model,
            request.app.state.bm25
        )
        result = [convert_rag_search(r) for r in result]
        _cache_add(f"NONLLM{body.query}", {"result": result})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/search_llm")
async def search_llm(body: SearchRequest, request: Request):
    cached = _cache_get(f"LLM{body.query}")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(
            perform_rag_search_llm,
            request.app.state.llm_model,
            request.app.state.embed_model,
            request.app.state.chunks,
            request.app.state.chunks_with_key,
            request.app.state.embeddings,
            body.query,
            request.app.state.rerank_model,
            request.app.state.bm25
        )
        result = convert_rag_search_llm(result)
        _cache_add(f"LLM{body.query}", {"result": result})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/check_answer")
async def check_answer(body: CheckAnswerRequest, request: Request):
    cached = _cache_get(f"QUEST{body.question_id}")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(
            perform_check_answer,
            int(body.question_id),
            request.app.state.rerank_model,
            request.app.state.question_dtos,
            body.user_answer,
            request.app.state.chunks_with_key
        )
        result = convert_check_answer(result)
        _cache_add(f"QUEST{body.question_id}", {"result": result})
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/questions")
async def get_question(request: Request):
    return {"questions": request.app.state.questions}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

    
