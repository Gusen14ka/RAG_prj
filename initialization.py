from pathlib import Path
import os
import torch

from entities import (
    RagState,
    Question
)
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
from parsing import pdf_to_chunks
from embedding import (
    build_model,
    build_and_save_embeddings,
    load_embeddings,
)
from keyword_search import (
    build_bm25,
    save_bm25,
    load_bm25
)
from utils.load import (
    load_json, 
    load_chunks_json_as_dict, 
    load_chunks_jsonl_as_list
)
from utils.save import (
    save_chunks_json_as_list,
    save_chunks_json_as_dict
)
from reranker import build_reranker
from questions_generator import generate_questions
from llm import Llm


def init_rag() -> RagState:
    os.makedirs("data", exist_ok=True)
    if not Path(CHUNK_PATH).exists():
        if not Path(SOURCE_PDF_PATH).exists():
            raise RuntimeError(f"Source PDF {SOURCE_PDF_PATH} not found. Please provide the PDF file.")
        
        chunks = pdf_to_chunks(SOURCE_PDF_PATH)
        save_chunks_json_as_list(chunks, CHUNK_PATH)
        save_chunks_json_as_dict(chunks, CHUNK_WITH_KEY_PATH)
        print(f"    Created {len(chunks)} chunks.")
    else:
        chunks = load_chunks_jsonl_as_list(CHUNK_PATH)
        save_chunks_json_as_dict(chunks, CHUNK_WITH_KEY_PATH)
        print(f"    Loaded {len(chunks)} chunks from cache.")
    chunks_with_key = load_chunks_json_as_dict(CHUNK_WITH_KEY_PATH)

    embed_model  = build_model(EMBADDING_MODEL_PATH)
    rerank_model = build_reranker(RERANK_MODEL_PATH)

    if not Path(EMBEDDINGS_PATH).exists():
        print("    Encoding chunks...")
        embeddings = build_and_save_embeddings(chunks, embed_model, EMBEDDINGS_PATH)
    else:
        embeddings = load_embeddings(EMBEDDINGS_PATH)

    if not Path(BM25_PATH).exists():
        bm25 = build_bm25(chunks)
        save_bm25(bm25, BM25_PATH)
    else:
        bm25 = load_bm25(BM25_PATH)

    print("    Loading LLM model...")
    llm_model = Llm(LLM_MODEL_PATH, gpu_mode=torch.cuda.is_available())
    print("    LLM susscesfully loaded")
    if not Path(QUESTIONS_BANK_PATH).exists():
        print("    Generating questions...") 
        generate_questions(SOURCE_PDF_PATH, embed_model, rerank_model, chunks, chunks_with_key, embeddings, llm_model, QUESTIONS_BANK_PATH)
    else:
        print("    Loading questions from file...")
        questions_dicts = load_json(QUESTIONS_BANK_PATH)
        question_dtos = [Question(**d) for d in questions_dicts]
        print(f"    Loaded {len(question_dtos)} questions.")

    return RagState(
        chunks=chunks,
        chunks_with_key=chunks_with_key,
        embed_model=embed_model,
        rerank_model=rerank_model,
        embeddings=embeddings,
        bm25=bm25,
        question_dtos=question_dtos,
        llm_model=llm_model
    )