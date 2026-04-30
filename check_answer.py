import re

from entities import (
    Question, 
    ResponseCheckAnswer,
    Chunk
)
from config.constants import ANSWER_THRESHOLD, Q_SCORE_WEIGHT, GOLD_SCORE_WEIGHT

MULTISPACE_RE = re.compile(r"\s+")

def _check_answer(rerank_model, answer: str, question: Question) -> tuple[bool, float]:
    """Проверяем ответ пользователя. Возвращает [правильно/неправильно, score]"""
    if not answer:
            return False, 0.0
    
    answer = MULTISPACE_RE.sub(" ", answer).strip()
    
    # Проводим сравнение рерак-моделью:
    q_score = rerank_model.predict([(answer, question.question)], show_progress_bar=False)[0]
    gold_score = sum([rerank_model.predict([(answer, question.answers[i])], show_progress_bar=False)[0] for i in range(len(question.answers))]) / len(question.answers)

    final_score = q_score * Q_SCORE_WEIGHT + gold_score * GOLD_SCORE_WEIGHT

    return final_score >= GOLD_SCORE_WEIGHT, final_score

def perform_check_answer(
        question_id: int,
        rerank_model,
        questions: list[Question],
        user_answer: str,
        chunks_with_key: dict[str, Chunk]
) -> ResponseCheckAnswer:
    question = next((q for q in questions if q.question_id == question_id), None)
    if not question:
        raise ValueError("Question not found")
    is_correct, score = _check_answer(rerank_model, user_answer, question)

    return ResponseCheckAnswer(is_correct, score, question.answers, [chunks_with_key[cid] for cid in question.chunk_ids])
      