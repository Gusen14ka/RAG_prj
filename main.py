import os
import sys
import argparse
from pathlib import Path

# UTF-8 вывод на Windows (cmd/PowerShell по умолчанию cp1251)
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure') and hasattr(sys.stderr, 'reconfigure'):
        sys.stdout.reconfigure(encoding="utf-8") #type: ignore
        sys.stderr.reconfigure(encoding="utf-8") #type: ignore

from parsing import pdf_to_chunks, save_chunks
from embedding import (
    load_chunks,
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
from utils.load_chunks import load_chunks_with_key
from reranker import rerank, build_reranker
from trainer_mode import Trainer

# ── Пути ────────────────────────────────────────────────────────────────
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


def generate_questions():
    """Генерируем вопросник для тренажёра."""
    print("[TRAINER] Инициализация генерации вопросов...")
    os.makedirs("data", exist_ok=True)
    
    # Загружаем чанки
    if not Path(CHUNK_PATH).exists() or not Path(CHUNK_WITH_KEY_PATH).exists():
        print("    ОШИБКА: чанки не найдены. Сначала запустите режим 'search'.")
        return
    
    chunks_with_key = load_chunks_with_key(CHUNK_WITH_KEY_PATH)
    print(f"    Загружено {len(chunks_with_key)} чанков.")
    
    # Загружаем модель
    rerank_model = build_reranker()
    print("    Модель загружена.")
    
    # Генерируем вопросы
    trainer = Trainer(rerank_model, QUESTIONS_BANK_PATH, ANSWER_THRESHOLD, Q_SCORE_WEIGHT, SRC_SCORE_WEIGHT, GOLD_SCORE_WEIGHT)
    questions = []
    
    print("    Генерируем вопросы из чанков...")
    for i, (chunk_id, chunk) in enumerate(chunks_with_key.items(), 1):
        generated = trainer.build_question_bank(chunk["clean_text"], chunk_id, min_score=0.65)
        questions.extend(generated)
        print(f"      [{i}/{len(chunks_with_key)}] {chunk_id}: +{len(generated)} вопросов")
    
    trainer.save_questions(questions)
    print(f"\n✓ Всего генерировано {len(questions)} вопросов.")
    print(f"✓ Сохранено: {QUESTIONS_BANK_PATH}")
    print(f"✓ Минимальный порог качества: min_score=0.65")
    
    # Запускаем тренировку
    if questions:
        print("\nНачинаем тренировку. Введите 'выход' для завершения.\n")
        
        for i, q in enumerate(questions, 1):
            print(f"Вопрос {i}/{len(questions)}: {q.question}")
            user_answer = input("Ваш ответ: ").strip()
            
            if user_answer.lower() in ("выход", "exit", "quit"):
                print("Выход из тренировки.")
                break
            
            # Проверяем ответ (простое сравнение, нормализованное)
            is_correct, score = trainer.check_answer(user_answer, q)
            
            if is_correct:
                out = "✓ Правильный ответ!"
            else:
                out = "✗ Неправильный ответ."
            
            print(out + f"Оценка ответа = {score}")
            print(f"Правильный ответ: {q.answer}")
            print(f"Исходное предложение: {q.source_sentence}")
            print("-" * 60)
            
            # Спрашиваем, продолжить ли
            cont = input("Продолжить? (y/n): ").strip().lower()
            if cont not in ("y", "yes", "да", "д"):
                print("Выход из тренировки.")
                break
            print()
    else:
        print("Вопросы не сгенерированы.")


def main_search():
    """Основной режим: RAG поиск с RRF и переранжированием."""
    print("[SEARCH] Инициализация системы поиска...")
    os.makedirs("data", exist_ok=True)

    # ── 1. Чанкинг ──────────────────────────────────────────────────────
    print("[1/5] Чанкинг PDF...")
    print("    Примечание: чтобы обработать новый PDF — удали папку data/")
    if not Path(CHUNK_PATH).exists() or not Path(CHUNK_WITH_KEY_PATH).exists():
        if not Path(SOURCE_PDF_PATH).exists():
            print(f"    ОШИБКА: {SOURCE_PDF_PATH} не найден.")
            return
         # Костыль против кривого парсинга
        if not Path(CHUNK_PATH).exists():
            chunks = pdf_to_chunks(SOURCE_PDF_PATH)
        else:
            chunks = load_chunks(CHUNK_PATH)
        save_chunks(chunks, CHUNK_PATH, CHUNK_WITH_KEY_PATH)
        print(f"    Создано {len(chunks)} чанков.")
    else:
        chunks = load_chunks(CHUNK_PATH)
        print(f"    Загружено {len(chunks)} чанков из кэша.")
    chunks_with_key = load_chunks_with_key(CHUNK_WITH_KEY_PATH)

    # ── 2. Загрузка моделей (до любых индексов) ──────────────────────────
    print("[2/5] Загрузка моделей...")
    embed_model  = build_model()
    rerank_model = build_reranker()
    print("    Модели загружены.")

    # ── 3. Векторные эмбеддинги (numpy, без faiss) ───────────────────────
    print("[3/5] Векторный индекс (numpy)...")
    if not Path(EMBEDDINGS_PATH).exists():
        print("    Кодируем чанки...")
        embeddings = build_and_save_embeddings(chunks, embed_model, EMBEDDINGS_PATH)
    else:
        embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"    Эмбеддингов: {embeddings.shape}")

    # ── 4. BM25 индекс ───────────────────────────────────────────────────
    print("[4/5] BM25 индекс...")
    if not Path(BM25_PATH).exists():
        print("    Строим BM25 индекс...")
        create_bm25_index_pipeline(CHUNK_PATH, BM25_PATH)
    bm25 = load_bm25(BM25_PATH)
    print("    BM25 загружен.")

    # ── 5. Цикл запросов ────────────────────────────────────────────────
    print("\n[5/5] Система готова. Введите 'выход' для завершения.\n")
    while True:

        query = input("Введите запрос: ").strip()
        if not query:
            continue
        if query.lower() in ("выход", "exit", "quit"):
            print("Выход.")
            break

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
            print("\n    Ничего не найдено выше порога релевантности.")
            print()
            continue

        print(f"\n{'─'*60}")
        print(f"Топ-{len(filtered)} результатов для: «{query}»")
        print(f"{'─'*60}")
        for i, (cid, score) in enumerate(filtered, 1):
            ch   = chunks_with_key[cid]
            text = ch.get("raw_text", "").strip()
            print(f"\n[{i}] chunk_id: {cid}  score={score:.4f}")
            print(text[:400])
            if len(text) > 400:
                print("    ...")
        print()


def main():
    """Главная точка входа с выбором режима."""
    parser = argparse.ArgumentParser(description="RAG система с тренажёром")
    parser.add_argument("mode", nargs="?", default="search", 
                       choices=["search", "trainer"],
                       help="Режим работы: 'search' (поиск) или 'trainer' (генерация вопросов и тренировка)")
    
    args = parser.parse_args()
    
    if args.mode == "trainer":
        generate_questions()
    elif args.mode == "search":
        main_search()


if __name__ == "__main__":
    main()