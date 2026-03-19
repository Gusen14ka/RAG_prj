from razdel import tokenize as razdel_tokenize
from pymorphy3 import MorphAnalyzer
from rank_bm25 import BM25Okapi
from utils.load_chunks import load_chunks
import pickle

morph = MorphAnalyzer()

def tokenize_lemmatize(text: str):
    tokens = [t.text for t in razdel_tokenize(text)]
    lemmas = []
    for tok in tokens:
        if tok.strip() == "":
            continue
        if tok.isalpha():
            p = morph.parse(tok)[0]
            lemmas.append(p.normal_form.lower())
    return lemmas

def build_bm25(chunks):
    texts = [c["raw_text"] for c in chunks]
    tokenized = [tokenize_lemmatize(t) for t in texts]

    bm25 = BM25Okapi(tokenized)

    return bm25, tokenized

def search(bm25, chunks, query, k=3):
    tokens = tokenize_lemmatize(query)

    score = bm25.get_scores(tokens)

    top_k = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:k]

    results = []

    for i in top_k:
        results.append({
            "score": float(score[i]),
            "chunk_id": chunks[i].get("chunk_id"),
            "raw_text": chunks[i].get("raw_text"),
            "metadata": chunks[i].get("metadata")
        })

    return results

def save_index(bm25, tokenized, bm25_path, bm25_tokenized_path):
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    with open(bm25_tokenized_path, "wb") as f:
        pickle.dump(tokenized, f)

    print("[SAVED] BM25 index")

def load_bm25(bm25_path):
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    return bm25

"""
Pipeline создания и сохранения bm25 и его индексов
"""
def create_bm25_index_pipeline(chunks_path = "data\\chunks.jsonl", bm25_path="data\\bm25.pkl", bm25_tokenized_path="data\\bm25_tokenized"):
    chunks = load_chunks(chunks_path)

    bm25, tokenized = build_bm25(chunks)

    save_index(bm25, tokenized, bm25_path, bm25_tokenized_path)

"""
Pipeline поиска по индексам bm25
"""
def search_bm25_pipeline(query: str, chunks_path = "data\\chunks.jsonl", bm25_path="data\\bm25.pkl", bm25_tokenized_path="data\\bm25_tokenized", top_k=3):
    chunks = load_chunks(chunks_path)

    bm25 = load_bm25(bm25_path)

    results = search(bm25, chunks, query, top_k)

    return results


if __name__ == "__main__":
    chunks = load_chunks("data\\chunks.jsonl")

    bm25, tokenized = build_bm25(chunks)

    res = search(bm25, chunks, "Что такое мультимножество", 5)

    for r in res:
            print("\n---")
            print("score:", r["score"])
            print("text:", r["chunk_id"])
