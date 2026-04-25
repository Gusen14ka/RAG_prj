import sys
import argparse

# UTF-8 вывод на Windows (cmd/PowerShell по умолчанию cp1251)
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure') and hasattr(sys.stderr, 'reconfigure'):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

from pipeline import (
    build_search_pipeline,
    QUESTIONS_BANK_PATH,
    CHUNK_WITH_KEY_PATH,
    CHUNK_PATH,
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
from utils.load_chunks import load_chunks, load_chunks_with_key
from reranker import rerank, build_reranker
from trainer_mode import Trainer, Question


def generate_questions():
    """Генерируем вопросник для тренажёра."""
    print("[TRAINER] Инициализация генерации вопросов...")

    if not CHUNK_PATH.exists() or not CHUNK_WITH_KEY_PATH.exists():
        print("    ОШИБКА: чанки не найдены. Сначала запустите режим 'search'.")
        return

    chunks_with_key = load_chunks_with_key(str(CHUNK_WITH_KEY_PATH))
    print(f"    Загружено {len(chunks_with_key)} чанков.")

    rerank_model = build_reranker()
    print("    Модель загружена.")

    trainer = Trainer(
        rerank_model, str(QUESTIONS_BANK_PATH),
        ANSWER_THRESHOLD, Q_SCORE_WEIGHT, SRC_SCORE_WEIGHT, GOLD_SCORE_WEIGHT,
    )
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

    if questions:
        print("\nНачинаем тренировку. Введите 'выход' для завершения.\n")

        for i, q in enumerate(questions, 1):
            print(f"Вопрос {i}/{len(questions)}: {q.question}")
            user_answer = input("Ваш ответ: ").strip()

            if user_answer.lower() in ("выход", "exit", "quit"):
                print("Выход из тренировки.")
                break

            is_correct, score = trainer.check_answer(user_answer, q)
            verdict = "✓ Правильный ответ!" if is_correct else "✗ Неправильный ответ."
            print(f"{verdict} Оценка = {score:.4f}")
            print(f"Правильный ответ: {q.answer}")
            print(f"Исходное предложение: {q.source_sentence}")
            print("-" * 60)

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

    state = build_search_pipeline(verbose=True)
    chunks          = state["chunks"]
    chunks_with_key = state["chunks_with_key"]
    embed_model     = state["embed_model"]
    rerank_model    = state["rerank_model"]
    embeddings      = state["embeddings"]
    bm25            = state["bm25"]

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

        res_rrf_ids = rrf_scoring([res_embedding, res_keyword], weight_list=[1, 1])
        res_rrf     = [chunks_with_key[x[0]] for x in res_rrf_ids]

        res_rerank_ids = rerank(query, res_rrf, TOP_K_RERANK, model=rerank_model)

        filtered = [(cid, score) for cid, score in res_rerank_ids
                    if score >= RERANK_THRESHOLD]

        if not filtered:
            print("\n    Ничего не найдено выше порога релевантности.\n")
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
    parser.add_argument(
        "mode", nargs="?", default="search",
        choices=["search", "trainer"],
        help="Режим работы: 'search' (поиск) или 'trainer' (генерация вопросов и тренировка)",
    )
    args = parser.parse_args()

    if args.mode == "trainer":
        generate_questions()
    else:
        main_search()


if __name__ == "__main__":
    main()
