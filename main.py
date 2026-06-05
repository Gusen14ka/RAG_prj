from __future__ import annotations

import random

from initialization import init_rag
from check_answer import perform_check_answer
from search_pipeline import perform_rag_search, perform_rag_search_llm


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_search_results(results) -> None:
    if not results:
        print("Ничего не найдено.")
        return

    for i, item in enumerate(results, start=1):
        chunk = item.chunk
        print(f"\n[{i}] chunk_id: {chunk.chunk_id}")
        print(f"score: {item.score:.4f}")
        print(f"section: {chunk.metadata.section}")
        print(f"subsection: {chunk.metadata.subsection}")
        print("text:")
        print(chunk.raw_text)
        print("-" * 80)


def _print_llm_result(result) -> None:
    print("\nОтвет LLM:")
    print(result.llm_answer)

    print("\nИспользованные источники:")
    for i, item in enumerate(result.chunks_response, start=1):
        chunk = item.chunk
        print(f"\n[{i}] chunk_id: {chunk.chunk_id}")
        print(f"score: {item.score:.4f}")
        print(f"section: {chunk.metadata.section}")
        print(f"subsection: {chunk.metadata.subsection}")
        print("text:")
        print(chunk.raw_text)
        print("-" * 80)


def _print_check_answer_result(result) -> None:
    print("\nРезультат проверки:")
    print(f"is_correct: {result.is_correct}")
    print(f"score: {result.score:.4f}")

    gold_answers = getattr(result, "gold_answers", None)
    if gold_answers:
        print("gold_answers:")
        for ans in gold_answers:
            print(f"- {ans}")

    source_sentence = getattr(result, "source_sentence", None)
    if source_sentence:
        print("\nsource_sentence:")
        print(source_sentence)

    gold_answer = getattr(result, "gold_answer", None)
    if gold_answer:
        print("\ngold_answer:")
        print(gold_answer)


def _choose_mode() -> str:
    while True:
        _print_header("RAG CLI")
        print("1 — Тренажёр")
        print("2 — Учебник")
        print("q — Выход")

        choice = input("\nВыбор: ").strip().lower()
        if choice in {"1", "2", "q"}:
            return choice

        print("Некорректный выбор.")


def _trainer_mode(state) -> None:
    if not state.question_dtos:
        print("Банк вопросов пуст.")
        input("Нажми Enter, чтобы вернуться в меню...")
        return

    while True:
        question = random.choice(state.question_dtos)

        _print_header("Тренажёр")
        print(f"question_id: {question.question_id}")
        print(f"term: {question.term}")
        print("\nВопрос:")
        print(question.question)

        print("\nПодсказка:")
        print("Напиши ответ и нажми Enter.")
        print("Введи /back, чтобы вернуться в меню.")
        print("Введи /next, чтобы сразу взять другой вопрос.")

        user_answer = input("\nТвой ответ: ").strip()
        if not user_answer:
            continue
        if user_answer.lower() == "/back":
            return
        if user_answer.lower() == "/next":
            continue

        result = perform_check_answer(
            int(question.question_id),
            state.rerank_model,
            state.question_dtos,
            user_answer,
            state.chunks_with_key,
        )

        _print_check_answer_result(result)

        cmd = input("\nEnter — следующий вопрос, /back — меню: ").strip().lower()
        if cmd == "/back":
            return


def _textbook_search_once(state, use_llm: bool) -> None:
    query = input("\nВведите запрос (или /back): ").strip()
    if not query:
        return
    if query.lower() == "/back":
        raise KeyboardInterrupt

    if use_llm:
        result = perform_rag_search_llm(
            state.llm_model,
            state.embed_model,
            state.chunks,
            state.chunks_with_key,
            state.embeddings,
            query,
            state.rerank_model,
            state.bm25,
        )
        _print_llm_result(result)
    else:
        result = perform_rag_search(
            state.embed_model,
            state.chunks,
            state.chunks_with_key,
            state.embeddings,
            query,
            state.rerank_model,
            state.bm25,
        )
        _print_search_results(result)


def _textbook_mode(state) -> None:
    while True:
        _print_header("Учебник")
        print("1 — Поиск без LLM")
        print("2 — Поиск с LLM")
        print("b — Назад в меню")

        choice = input("\nВыбор: ").strip().lower()
        if choice == "b":
            return
        if choice not in {"1", "2"}:
            print("Некорректный выбор.")
            continue

        use_llm = choice == "2"

        while True:
            try:
                _textbook_search_once(state, use_llm=use_llm)
            except KeyboardInterrupt:
                break

            cmd = input("\nEnter — ещё один запрос, /mode — сменить режим, /back — меню: ").strip().lower()
            if cmd == "/back":
                return
            if cmd == "/mode":
                break


def main() -> None:
    state = init_rag()

    while True:
        choice = _choose_mode()

        if choice == "q":
            print("Выход.")
            return

        try:
            if choice == "1":
                _trainer_mode(state)
            elif choice == "2":
                _textbook_mode(state)
        except KeyboardInterrupt:
            print("\nВозврат в меню.")


if __name__ == "__main__":
    main()