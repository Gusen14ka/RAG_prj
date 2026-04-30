from dataclasses import dataclass
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)
from rank_bm25 import BM25Okapi

from llm import Llm

@dataclass
class Question:
    question_id: int
    term: str
    question: str
    answers: list[str]
    chunk_ids: list[str]

@dataclass
class ChunkMetaData:
    section: str
    subsection: str

@dataclass
class PageContrainer:
    section_id: str
    page_texts: list[str]
    metadata: ChunkMetaData

@dataclass
class Chunk:
    chunk_id: str
    clean_text: str
    raw_text: str
    metadata: ChunkMetaData

@dataclass
class ResponseRagSearch:
    chunk: Chunk
    score: float

@dataclass
class ResponseRagSearchLlm:
    chunks_response: list[ResponseRagSearch]
    llm_answer: str

@dataclass
class ResponseCheckAnswer:
    is_correct: bool
    score: float
    gold_answers: list[str]
    source_chunks: list[Chunk]

@dataclass
class RagState:
    chunks: list[Chunk]
    chunks_with_key: dict[str, Chunk]
    embed_model: SentenceTransformer
    rerank_model: CrossEncoder
    embeddings: np.ndarray
    bm25: BM25Okapi
    question_dtos: list[Question]
    llm_model: Llm