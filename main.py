from pathlib import Path
from parsing import pdf_to_chunks, save_chunks
from embedding import load_chunks, build_and_save_faiss_index, load_index_and_metadata, search
from keyword_search import create_bm25_index_pipeline, search_bm25_pipeline
from utils.rrf_scoring import rrf_scoring
from utils.load_chunks import load_chunks_with_key
from reranker import rerank

CHUNK_PATH = "data/chunks.jsonl"
CHUNK_WITH_KEY_PATH = "data/chunks_with_key.json"
SOURCE_PDF_PATH = "test.pdf"
VECTOR_DB_PATH = "data/faiss_index.ivf"
BM25_PATH = "data/bm25.pkl"
TOP_K_FOR_BM_AND_EMD = 5
TOP_K_FOR_RERANK = 3
"""
Основной pipeline приложения
"""
def main():
    # Chunking
    chunks = []
    chunks_with_key = []
    chunks_file = Path(CHUNK_PATH)
    chunks_with_key_file = Path(CHUNK_WITH_KEY_PATH)
    if not chunks_file.exists() or not chunks_with_key_file.exists():
        chunks = pdf_to_chunks(SOURCE_PDF_PATH)
        save_chunks(chunks, CHUNK_PATH, CHUNK_WITH_KEY_PATH)
        chunks_with_key = load_chunks_with_key(CHUNK_WITH_KEY_PATH)
    else:
        chunks = load_chunks(CHUNK_PATH)
        chunks_with_key = load_chunks_with_key(CHUNK_WITH_KEY_PATH)
    
    # Embedding - подготавливаем индексы векторной бд
    vector_db_file = Path(VECTOR_DB_PATH)
    if not vector_db_file.exists():
        build_and_save_faiss_index(chunks, index_path=VECTOR_DB_PATH)
    
    index = load_index_and_metadata(VECTOR_DB_PATH)

    # Keyword search - подготавливаем bm25
    bm25_file = Path(BM25_PATH)
    if not bm25_file.exists():
        create_bm25_index_pipeline(CHUNK_PATH, BM25_PATH)

    # Пользователь вводит запрос
    query = "Приведи пример конканетации"

    # Ищем по эмбеддингу
    res_embedding = search(index, chunks, query, top_k=TOP_K_FOR_BM_AND_EMD)

    # Ищем по keyword search
    res_keyword = search_bm25_pipeline(query, CHUNK_PATH, BM25_PATH, top_k=TOP_K_FOR_BM_AND_EMD)

    # Делаем RRF
    res_rrf_ids = rrf_scoring([res_embedding, res_keyword], weight_list=[1, 1])
    res_rrf = [chunks_with_key[x[0]] for x in res_rrf_ids]

    # Rerank
    res_rerank_ids = rerank(query, res_rrf, TOP_K_FOR_RERANK)
    final_result = [chunks_with_key[x[0]] for x in res_rerank_ids]

    print("___RERANK___")
    for ch in final_result:
        print(ch)
        print()




if __name__ == "__main__":
    main()