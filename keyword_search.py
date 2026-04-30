from razdel import tokenize as razdel_tokenize
from pymorphy3 import MorphAnalyzer
from rank_bm25 import BM25Okapi
import pickle

from utils.load import load_chunks_jsonl_as_list
from entities import Chunk

_morph: MorphAnalyzer | None = None


def _get_morph() -> MorphAnalyzer:
    global _morph
    if _morph is None:
        _morph = MorphAnalyzer()
    return _morph


def tokenize_lemmatize(text: str):
    morph = _get_morph()
    tokens = [t.text for t in razdel_tokenize(text)]
    lemmas = []
    for tok in tokens:
        if tok.strip() == "":
            continue
        if tok.isalpha():
            p = morph.parse(tok)[0]
            lemmas.append(p.normal_form.lower())
    return lemmas


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    texts = [c.raw_text for c in chunks]
    tokenized = [tokenize_lemmatize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25


def search(bm25, chunks: list[Chunk], query: str, k=3):
    tokens = tokenize_lemmatize(query)
    score = bm25.get_scores(tokens)
    top_k = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:k]
    results = []
    for i in top_k:
        results.append(chunks[i])
    return results


def search_bm25(bm25, chunks: list[Chunk], query: str, top_k: int = 3) -> list[Chunk]:
    """Поиск с уже загруженными индексом и чанками (без I/O)."""
    return search(bm25, chunks, query, top_k)


def save_bm25(bm25, bm25_path: str):
    """Сохраняет только BM25 объект — tokenized не нужен для поиска."""
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print("[SAVED] BM25 index")


def load_bm25(bm25_path: str):
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    return bm25


def create_bm25_index_pipeline(
    chunks_path="data/chunks.jsonl",
    bm25_path="data/bm25.pkl",
):
    """Pipeline создания и сохранения BM25 индекса."""
    chunks = load_chunks_jsonl_as_list(chunks_path)
    bm25 = build_bm25(chunks)
    save_bm25(bm25, bm25_path)


def search_bm25_pipeline(
    query: str,
    chunks_path="data/chunks.jsonl",
    bm25_path="data/bm25.pkl",
    top_k=3,
):
    """Pipeline поиска (загружает всё с диска — используй search_bm25 если уже загружено)."""
    chunks = load_chunks_jsonl_as_list(chunks_path)
    bm25 = load_bm25(bm25_path)
    return search(bm25, chunks, query, top_k)


