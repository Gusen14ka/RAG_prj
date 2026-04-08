from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import os
import uvicorn

from parsing import pdf_to_chunks, save_chunks
from embedding import (
    build_model,
    build_and_save_embeddings,
    load_embeddings,
    search as embedding_search,
)
from keyword_search import (
    create_bm25_index_pipeline,
    load_bm25,
    search_bm25,
)
from utils.rrf_scoring import rrf_scoring
from utils.load_chunks import load_chunks_with_key, load_chunks
from reranker import rerank, build_reranker
from trainer_mode import Trainer, Question

CHUNK_PATH          = "data/chunks.jsonl"
CHUNK_WITH_KEY_PATH = "data/chunks_with_key.json"
SOURCE_PDF_PATH     = "test.pdf"
EMBEDDINGS_PATH     = "data/embeddings.npy"
BM25_PATH           = "data/bm25.pkl"
QUESTIONS_BANK_PATH = "data/questions_bank.jsonl"

TOP_K_RETRIEVAL  = 5
TOP_K_RERANK     = 3
RERANK_THRESHOLD = 0.3   # чанки ниже этого score не показываем

ANSWER_THRESHOLD = 0.3
Q_SCORE_WEIGHT = 0.35
SRC_SCORE_WEIGHT = 0.45
GOLD_SCORE_WEIGHT = 0.20


@asynccontextmanager
async def lifecycle(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    if not Path(CHUNK_PATH).exists() or not Path(CHUNK_WITH_KEY_PATH).exists():
        if not Path(SOURCE_PDF_PATH).exists():
            raise RuntimeError(f"Source PDF {SOURCE_PDF_PATH} not found. Please provide the PDF file.")
        chunks = pdf_to_chunks(SOURCE_PDF_PATH)
        save_chunks(chunks, CHUNK_PATH, CHUNK_WITH_KEY_PATH)
        print(f"    Created {len(chunks)} chunks.")
    else:
        chunks = load_chunks(CHUNK_PATH)
        print(f"    Loaded {len(chunks)} chunks from cache.")
    chunks_with_key = load_chunks_with_key(CHUNK_WITH_KEY_PATH)

    embed_model  = build_model()
    rerank_model = build_reranker()

    if not Path(EMBEDDINGS_PATH).exists():
        print("    Encoding chunks...")
        embeddings = build_and_save_embeddings(chunks, embed_model, EMBEDDINGS_PATH)
    else:
        embeddings = load_embeddings(EMBEDDINGS_PATH)

    if not Path(BM25_PATH).exists():
        create_bm25_index_pipeline(CHUNK_PATH, BM25_PATH)
    bm25 = load_bm25(BM25_PATH)

    trainer = Trainer(rerank_model, QUESTIONS_BANK_PATH, ANSWER_THRESHOLD, Q_SCORE_WEIGHT, SRC_SCORE_WEIGHT, GOLD_SCORE_WEIGHT)
    questions = []
    if not Path(QUESTIONS_BANK_PATH).exists():
        print("    Generating questions...")
        for i, (chunk_id, chunk) in enumerate(chunks_with_key.items(), 1):
            generated = trainer.build_question_bank(chunk["clean_text"], chunk_id, min_score=0.65)
            questions.extend(generated)
        print(f"    Generated {len(questions)} questions.")
        trainer.save_questions(questions)
    else:
        print("    Loading questions from file...")
        questions_dicts = load_chunks(QUESTIONS_BANK_PATH)
        questions = [Question(**d) for d in questions_dicts]
        print(f"    Loaded {len(questions)} questions.")

    # Сохраняем все необходимые объекты
    app.state.chunks = chunks
    app.state.chunks_with_key = chunks_with_key
    app.state.embed_model = embed_model
    app.state.rerank_model = rerank_model
    app.state.embeddings = embeddings
    app.state.bm25 = bm25
    app.state.trainer = trainer
    app.state.questions = questions

    print("    Lifespan complete.")

    yield

def perform_search(
        embed_model,
        rerank_model,
        bm25,
        chunks,
        chunks_with_key,
        embeddings,
        query
):
    res_embedding = embedding_search(embeddings, chunks, query,
                                         model=embed_model, top_k=TOP_K_RETRIEVAL)
    res_keyword   = search_bm25(bm25, chunks, query, top_k=TOP_K_RETRIEVAL)

    res_rrf_ids  = rrf_scoring([res_embedding, res_keyword], weight_list=[1, 1])
    res_rrf      = [chunks_with_key[x[0]] for x in res_rrf_ids]

    res_rerank_ids = rerank(query, res_rrf, TOP_K_RERANK, model=rerank_model)

    # Фильтр по порогу: убираем нерелевантные чанки
    filtered = [(cid, score) for cid, score in res_rerank_ids
                if score >= RERANK_THRESHOLD]
    
    if not filtered:
        return []
    
    res = []
    for i, (cid, score) in enumerate(filtered, 1):
        ch   = chunks_with_key[cid]
        text = ch.get("raw_text", "").strip()
        res.append({"text": text, "chunk_id": cid, "score": float(score)})

    return res

def perform_check_answer(
        question_id,
        trainer,
        questions,
        user_answer
):
    question = next((q for q in questions if q.question_id == question_id), None)
    if not question:
        return {"error": "Question not found"}
    
    is_correct, score = trainer.check_answer(user_answer, question)

    return {
        "is_correct": bool(is_correct),
        "score": float(score),
        "source_sentence": question.source_sentence,
        "gold_answer": question.answer,
    }



app = FastAPI(title="NON LLM RAG", version="1.0.0", lifespan=lifecycle)

class SearchRequest(BaseModel):
    query: str

class CheckAnswerRequest(BaseModel):
    question_id: str
    user_answer: str

@app.get("/health")
def health():
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
            body.query
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
            body.user_answer
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

    
