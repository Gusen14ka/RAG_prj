import os
from pathlib import Path

BASE_DIR = os.getenv("WORK_DIR", str(Path(__file__).resolve().parents[1]))


CHUNK_PATH          = BASE_DIR + "/data/chunks.jsonl"
CHUNK_WITH_KEY_PATH = BASE_DIR + "/data/chunks_with_key.json"
SOURCE_PDF_PATH     = BASE_DIR + "/doc.pdf"
EMBEDDINGS_PATH     = BASE_DIR + "/data/embeddings.npy"
BM25_PATH           = BASE_DIR + "/data/bm25.pkl"
QUESTIONS_BANK_PATH = BASE_DIR + "/data/questions_bank.json"
EMBADDING_MODEL_PATH = BASE_DIR + "/models/multilingual-e5-large"
RERANK_MODEL_PATH = BASE_DIR + "/models/bge-reranker-v2-m3"
LLM_MODEL_PATH = BASE_DIR + "/models/qwen2.5-3b"