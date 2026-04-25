from dataclasses import dataclass, asdict
import json
import re
from typing import List, Optional, Any, Sequence, Tuple, Iterable, Dict
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)
import yake


MULTISPACE_RE = re.compile(r"\s+")
HEADER_RE = re.compile(r"^(?:Раздел:|Подраздел:|Примечание:)\s+.+$", flags=re.MULTILINE)

BLACKLIST_WORDS = {
    "раздел", "подраздел", "примечание", "параграф", "введение", "заключение",
    "содержание", "предисловие", "изложение", "сущность", "основной", "главный",
}

# ─── Утилиты текста ───────────────────────────────────────────────────────────

def _capitalize(s: str) -> str:
    return (s[0].upper() + s[1:]) if s else s


def normalize_spaces(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def clean_text_for_nlp(text: str) -> str:
    """Нормализация для NLP (natasha): сохраняем оригинальный регистр."""
    text = text.replace("\u00ad", "")   # мягкий дефис
    text = text.replace("ё", "е")
    return normalize_spaces(text)


def clean_text_for_check(text: str) -> str:
    """Нормализация для проверки ответа: lowercase + нормализация."""
    text = text.replace("\u00ad", "")
    text = text.lower().strip()
    text = text.replace("ё", "е")
    return normalize_spaces(text)


# Оставляем псевдоним для обратной совместимости (check_answer его использует)
def clean_text(text: str) -> str:
    return clean_text_for_check(text)


def remove_headers(text: str) -> str:
    """Удаляем строки 'Раздел: ...', 'Подраздел: ...' и нумерацию в начале."""
    lines = text.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (HEADER_RE.match(stripped) or
                re.match(r"^(?:№\s*)?\d+(?:\.\d+)*\.?\s+", stripped) or
                stripped == ""):
            start_idx = i + 1
        else:
            break

    clean_lines = lines[start_idx:]
    if clean_lines:
        first_line = clean_lines[0].strip()
        if (1 < len(first_line.split()) <= 6 and
                not re.search(r"[.!?;:—–-]", first_line) and
                re.match(r"^[А-ЯЁ][А-Яа-яё\s]+$", first_line)):
            clean_lines = clean_lines[1:]

    text = '\n'.join(clean_lines).strip()
    text = re.sub(r'\s+[А-ЯЁ][А-Яа-яё\s-]*[∈∅]\s*—[\s\S]*', '', text)
    text = re.sub(r'^(?:[А-ЯЁ][а-яё]+(?:\s+[а-яё]+){0,4})\s+(?=[А-ЯЁ][а-яё]*\b)', '', text)
    text = re.sub(r"№\s*\d+(?:\.\d+)*\.?\s*", "", text)
    text = re.sub(r"(?m)^\s*\d+(?:\.\d+)*\.?\s*", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)*\.?\s+(?=[А-ЯЁ])", "", text)
    text = re.sub(r'Название\s+параграфа\s+Ключевые\s+термины\s+и\s+обозначения', '', text)
    text = re.sub(r'Ключевые\s+термины\s+и\s+обозначения', '', text)
    text = re.sub(r'Название\s+параграфа', '', text)

    lines_clean = []
    for line in text.split('\n'):
        stripped = line.strip()
        if re.match(r"^\s*\d+(?:\.\d+)*\.?\s+[А-ЯЁ]", stripped):
            continue
        if (1 < len(stripped.split()) <= 6 and
                not re.search(r"[.!?;:—–-]", stripped) and
                re.match(r"^[А-ЯЁ][^а-яё]*$", stripped)):
            continue
        lines_clean.append(line)

    return '\n'.join(lines_clean).strip()


# ─── NLP утилиты ─────────────────────────────────────────────────────────────

def get_lemma(token: Any) -> str:
    lemma = getattr(token, "lemma", None)
    if lemma:
        return str(lemma).lower()
    return str(getattr(token, "text", "")).lower()


def sentence_lemmas(sentence: Any) -> List[str]:
    return [get_lemma(t) for t in sentence.tokens]


def has_any_lemma(lemmas: Sequence[str], candidates: Sequence[str]) -> bool:
    s = set(lemmas)
    return any(c in s for c in candidates)


def find_first_fragment(sentence: str, fragment: str) -> Optional[Tuple[int, int]]:
    if not fragment or len(fragment) < 2:
        return None
    idx = sentence.lower().find(fragment.lower())
    if idx < 0:
        return None
    return idx, idx + len(fragment)


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def is_valid_question(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    if '?' not in question:
        return False
    if len(question.split()) < 3:
        return False
    if '...' in answer or '…' in answer:
        return False
    words = answer.split()
    if words and words[0].lower() in BLACKLIST_WORDS:
        return False
    if not re.search(r'[а-яё]', answer.lower()):
        return False
    return True


# ─── Dataclass ───────────────────────────────────────────────────────────────

@dataclass
class Question:
    question_id: str
    question: str
    answer: str
    keywords: List[str]
    source_chunk_id: str
    source_sentence: str
    score: float


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
            self,
            rerank_model,
            question_bank_path: str,
            answer_threshold: float,
            q_score_weight: float,
            src_score_weight: float,
            gold_score_weight: float) -> None:
        self.question_bank_path = question_bank_path
        self.rerank_model       = rerank_model
        self.answer_threshold   = answer_threshold
        self.q_score_weight     = q_score_weight
        self.src_score_weight   = src_score_weight
        self.gold_score_weight  = gold_score_weight

        self.segmenter    = Segmenter()
        self.morph_vocab  = MorphVocab()
        self.emb          = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger   = NewsNERTagger(self.emb)

        self.kw_extractor = yake.KeywordExtractor(
            lan="ru", n=3, top=7, dedupLim=0.9, windowsSize=2
        )

    # ── Сохранение ───────────────────────────────────────────────────────────

    def save_questions(self, questions: List[Question]) -> None:
        with open(self.question_bank_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    # ── Ключевые слова ────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> List[str]:
        if len(text.split()) < 3:
            return []
        items = self.kw_extractor.extract_keywords(text)
        keywords = []
        for phrase, _ in items:
            phrase = phrase.strip().strip(",.;:!?")
            if phrase and len(phrase.split()) <= 3:
                keywords.append(phrase)
        return dedupe_preserve(keywords)[:5]

    # ── Классификация предложений ─────────────────────────────────────────────

    def _classify_sentence(self, sentence: Any) -> Optional[str]:
        lemmas = sentence_lemmas(sentence)
        text   = sentence.text.lower()

        # Определения — самый надёжный тип
        if has_any_lemma(lemmas, [
            "это", "называться", "являться", "представлять", "быть",
            "определять", "обозначать", "означать", "понимать", "подразумевать",
        ]):
            return "definition"

        # Состав / включение
        if has_any_lemma(lemmas, [
            "включать", "состоять", "содержать", "входить", "состав",
            "объединять", "разбиваться", "делиться",
        ]):
            return "composition"

        # Использование / назначение
        if has_any_lemma(lemmas, [
            "использовать", "применять", "служить", "предназначаться",
            "позволять", "обеспечивать",
        ]):
            return "usage"

        # Свойства / характеристики
        if has_any_lemma(lemmas, [
            "характеризовать", "характеризоваться", "свойство",
            "обладать", "иметь", "принадлежать",
        ]):
            return "property"

        # Зависимости (убрана лемма "как" — слишком широкая)
        if has_any_lemma(lemmas, ["зависеть", "влиять", "связь"]):
            return "dependency"

        # Сравнение
        if "в отличие" in text or has_any_lemma(
            lemmas, ["отличаться", "сравнить", "различие", "подобно", "схожий"]
        ):
            return "comparison"

        # Причина / следствие
        if any(word in text for word in [
            "потому что", "поэтому", "причина", "вследствие", "из-за", "так как"
        ]):
            return "cause"

        return None

    # ── Генерация вопроса ─────────────────────────────────────────────────────

    def _generate_question(
            self, sentence: str, kind: str, doc: Any
    ) -> Tuple[str, str, float]:
        """Генерируем пару (вопрос, ответ, confidence). При неудаче возвращаем ('','',0.0)."""
        s = normalize_spaces(sentence)
        s = re.sub(r"^\d+\.\d+(\.\d+)?\s+", "", s).strip()

        def _try_patterns(patterns, q_template_fn, max_ans_words: int = 20):
            for pattern, conf in patterns:
                m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
                if not m:
                    continue
                try:
                    groups = [g.strip().strip(",.;:-") for g in m.groups() if g]
                    if len(groups) < 2:
                        continue
                    subject, answer = groups[0], groups[-1]
                    subject = re.sub(r"^\d+\.\d+(\.\d+)?\s+", "", subject).strip()
                    if not subject or len(subject) < 2 or len(answer) < 3:
                        continue
                    if len(answer.split()) > max_ans_words:
                        continue
                    question = q_template_fn(subject, answer, groups)
                    if not question or not is_valid_question(question, answer):
                        continue
                    return _capitalize(question), _capitalize(answer), conf
                except Exception:
                    continue
            return "", "", 0.0

        # ── Определения ──────────────────────────────────────────────────────
        if kind == "definition":
            patterns = [
                # "X — это Y" / "X — Y"
                (r"^(.+?)\s*(?:—|–|-)\s*это\s+(.+?)(?:[.;!?]|$)", 0.95),
                # "X есть Y"
                (r"^(.+?)\s+есть\s+(.+?)(?:[.;!?]|$)", 0.90),
                # "X это Y" (без тире)
                (r"^(.+?)\s+это\s+(.+?)(?:[.;!?]|$)", 0.88),
                # "X называется Y"
                (r"^(.+?)\s+называется\s+(.+?)(?:[.;!?]|$)", 0.85),
                # "X называют Y"
                (r"^(.+?)\s+называют\s+(.+?)(?:[.;!?]|$)", 0.85),
                # "X определяется как Y"
                (r"^(.+?)\s+определяется\s+(?:как\s+)?(.+?)(?:[.;!?]|$)", 0.88),
                # "Под X понимают/понимается Y"
                (r"^(?:под\s+)?(.+?)\s+понимают\s+(.+?)(?:[.;!?]|$)", 0.87),
                (r"^(?:под\s+)?(.+?)\s+понимается\s+(.+?)(?:[.;!?]|$)", 0.87),
                # "X обозначает/означает Y"
                (r"^(.+?)\s+обозначает\s+(.+?)(?:[.;!?]|$)", 0.83),
                (r"^(.+?)\s+означает\s+(.+?)(?:[.;!?]|$)", 0.83),
                # "Термином X называют Y"
                (r"термином?\s+[«\"]?(.+?)[»\"]?\s+называют\s+(.+?)(?:[.;!?]|$)", 0.90),
                # "Термин X используется для обозначения Y"
                (r"термин\s+[«\"]?(.+?)[»\"]?\s+(?:используется\s+)?для\s+обозначения\s+(.+?)(?:[.;!?]|$)", 0.92),
            ]

            def def_q(subject, answer, groups):
                # Не оставляем предлоги в начале субъекта
                if subject.split()[0].lower() in {"при", "в", "на", "по", "для", "из", "с", "через", "со", "от", "под"}:
                    return ""
                return f"Что такое {subject}?"

            return _try_patterns(patterns, def_q, max_ans_words=22)

        # ── Состав ───────────────────────────────────────────────────────────
        elif kind == "composition":
            patterns = [
                (r"^(.+?)\s+состоит\s+из\s+(.+?)(?:[.;!?]|$)", 0.92),
                (r"^(.+?)\s+включает\s+(?:в\s+себя\s+)?(.+?)(?:[.;!?]|$)", 0.88),
                (r"^(.+?)\s+содержит\s+(.+?)(?:[.;!?]|$)", 0.85),
                (r"^(?:в\s+состав\s+)?(.+?)\s+входят?\s+(.+?)(?:[.;!?]|$)", 0.83),
            ]

            def comp_q(subject, answer, groups):
                if "состоит" in s.lower():
                    return f"Из чего состоит {subject}?"
                if "содержит" in s.lower():
                    return f"Что содержит {subject}?"
                return f"Что включает {subject}?"

            return _try_patterns(patterns, comp_q, max_ans_words=20)

        # ── Использование ─────────────────────────────────────────────────────
        elif kind == "usage":
            patterns = [
                (r"^(.+?)\s+используется\s+(?:в\s+|для\s+)?(.+?)(?:[.;!?]|$)", 0.88),
                (r"^(.+?)\s+применяется\s+(?:в\s+|для\s+)?(.+?)(?:[.;!?]|$)", 0.85),
                (r"^(.+?)\s+служит\s+(?:для\s+)?(.+?)(?:[.;!?]|$)", 0.85),
                (r"^(.+?)\s+предназначен(?:а|о|ы)?\s+(?:для\s+)?(.+?)(?:[.;!?]|$)", 0.83),
                (r"^(.+?)\s+позволяет\s+(.+?)(?:[.;!?]|$)", 0.82),
            ]

            def usage_q(subject, answer, groups):
                return f"Для чего используется {subject}?"

            return _try_patterns(patterns, usage_q, max_ans_words=18)

        # ── Свойства ──────────────────────────────────────────────────────────
        elif kind == "property":
            patterns = [
                (r"^(.+?)\s+определяется\s+(?:как\s+)?(.+?)(?:[.;!?]|$)", 0.87),
                (r"^(.+?)\s+характеризуется\s+(.+?)(?:[.;!?]|$)", 0.82),
                (r"^(.+?)\s+обладает\s+(.+?)(?:[.;!?]|$)", 0.80),
                (r"^(?:свойством\s+)?(.+?)\s+является\s+(.+?)(?:[.;!?]|$)", 0.80),
                (r"^(.+?)\s+имеет\s+(.+?)(?:[.;!?]|$)", 0.75),
            ]

            def prop_q(subject, answer, groups):
                if "характеризуется" in s.lower():
                    return f"Чем характеризуется {subject}?"
                if "обладает" in s.lower():
                    return f"Каким свойством обладает {subject}?"
                return f"Какое свойство имеет {subject}?"

            return _try_patterns(patterns, prop_q, max_ans_words=16)

        # ── Зависимости ───────────────────────────────────────────────────────
        elif kind == "dependency":
            patterns = [
                (r"^(.+?)\s+зависит\s+от\s+(.+?)(?:[.;!?]|$)", 0.85),
                (r"^(.+?)\s+влияет\s+на\s+(.+?)(?:[.;!?]|$)", 0.83),
                (r"^(.+?)\s+определяется\s+(.+?)(?:[.;!?]|$)", 0.80),
            ]

            def dep_q(subject, answer, groups):
                if "зависит" in s.lower():
                    return f"От чего зависит {subject}?"
                if "влияет" in s.lower():
                    return f"На что влияет {subject}?"
                return f"Чем определяется {subject}?"

            return _try_patterns(patterns, dep_q, max_ans_words=15)

        # ── Сравнение ─────────────────────────────────────────────────────────
        elif kind == "comparison":
            patterns = [
                # "X отличается от Y тем, что Z"
                (r"^(.+?)\s+отличается\s+от\s+(.+?)\s+тем[,\s]+(?:что\s+)?(.+?)(?:[.;!?]|$)", 0.85),
                # "X отличается от Y + Z"
                (r"^(.+?)\s+отличается\s+от\s+(.+?)(?:[.;!?]|$)", 0.78),
            ]
            for pattern, conf in patterns:
                m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
                if not m:
                    continue
                try:
                    groups = [g.strip().strip(",.;:-") for g in m.groups() if g]
                    if len(groups) < 2:
                        continue
                    subject  = groups[0]
                    compared = groups[1]
                    answer   = groups[2] if len(groups) > 2 else compared
                    if len(answer) < 3 or len(answer.split()) > 18:
                        continue
                    question = f"Чем {_capitalize(subject)} отличается от {compared}?"
                    if not is_valid_question(question, answer):
                        continue
                    return question, _capitalize(answer), conf
                except Exception:
                    continue

        # ── Причина ───────────────────────────────────────────────────────────
        elif kind == "cause":
            patterns = [
                (r"^(.+?)\s+потому\s+что\s+(.+?)(?:[.;!?]|$)", 0.83),
                (r"^(.+?)\s+так\s+как\s+(.+?)(?:[.;!?]|$)", 0.80),
                (r"^(.+?)\s+вследствие\s+(.+?)(?:[.;!?]|$)", 0.78),
            ]

            def cause_q(subject, answer, groups):
                return f"Почему {subject}?"

            return _try_patterns(patterns, cause_q, max_ans_words=18)

        return "", "", 0.0

    # ── Оценка качества ───────────────────────────────────────────────────────

    def _score_item(
            self, qtype: str, sentence: str, question: str,
            answer: str, keywords: Sequence[str], conf: float
    ) -> float:
        base_scores = {
            "definition": 1.0,
            "composition": 0.9,
            "usage":       0.88,
            "property":    0.85,
            "comparison":  0.83,
            "dependency":  0.80,
            "cause":       0.78,
        }
        base = base_scores.get(qtype, 0.0) * conf

        penalty = 0.0
        sent_len = len(sentence.split())
        if sent_len > 70:
            penalty += 0.35
        elif sent_len > 45:
            penalty += 0.15

        ans_len = len(answer.split())
        if ans_len > 18:
            penalty += 0.25
        elif ans_len > 12:
            penalty += 0.10

        if not keywords:
            penalty += 0.15

        bonus = 0.0
        if answer.lower() in sentence.lower():
            bonus += 0.15
        if question.endswith("?") and len(question.split()) >= 3:
            bonus += 0.10

        return round(max(0.0, min(1.0, base + bonus - penalty)), 4)

    # ── Основная генерация банка вопросов ────────────────────────────────────

    def build_question_bank(
            self, text: str, chunk_id: str, min_score: float = 0.55
    ) -> List[Question]:
        """
        Генерируем вопросы из чанка.

        Изменения относительно оригинала:
        - clean_text_for_nlp() вместо clean_text() — не лоукейсим
        - Диагностические счётчики
        - min_score=0.55 (было 0.65)
        """
        text = remove_headers(text)

        first_line = text.split('\n')[0].strip() if text else ""
        if first_line and re.match(r"^[А-ЯЁ][А-Яа-яЁё\s]+$", first_line) and len(first_line.split()) <= 5:
            text = '\n'.join(text.split('\n')[1:]).strip()

        text = re.sub(r'(?m)^\s*\d+(?:\.\d+)*\.?\s*', '', text)
        text = re.sub(r'№\s*\d+(?:\.\d+)*\.?\s*', '', text)

        # ИСПРАВЛЕНО: не лоукейсим — наташа работает лучше на оригинальном регистре
        text = clean_text_for_nlp(text)

        out: List[Question] = []
        question_idx = 1

        # ── Диагностика ───────────────────────────────────────────────────────
        stats: Dict[str, int] = {
            "total": 0, "short": 0, "long": 0,
            "no_type": 0, "no_pattern": 0, "invalid": 0,
            "low_score": 0, "ok": 0,
        }
        type_counts: Dict[str, int] = {}

        doc = Doc(text)
        doc.segment(self.segmenter)

        if not doc.sents or not doc.tokens:
            return out

        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)

        if doc.spans:
            for span in doc.spans:
                try:
                    span.normalize(self.morph_vocab)
                except Exception:
                    pass

        for sentence in doc.sents:
            sent_words = len(sentence.text.split())
            stats["total"] += 1

            if sent_words < 5:
                stats["short"] += 1
                continue
            if sent_words > 100:
                stats["long"] += 1
                continue

            qtype = self._classify_sentence(sentence)
            if qtype is None:
                stats["no_type"] += 1
                continue

            type_counts[qtype] = type_counts.get(qtype, 0) + 1

            question, answer, regex_conf = self._generate_question(sentence.text, qtype, doc)
            if not question or not answer or regex_conf == 0.0:
                stats["no_pattern"] += 1
                continue

            if not is_valid_question(question, answer):
                stats["invalid"] += 1
                continue

            keywords = self._extract_keywords(sentence.text)
            score = self._score_item(qtype, sentence.text, question, answer, keywords, regex_conf)

            if score < min_score:
                stats["low_score"] += 1
                continue

            out.append(Question(
                question_id=f"{chunk_id}__q_{question_idx:05d}",
                question=question,
                answer=answer,
                keywords=keywords,
                source_chunk_id=chunk_id,
                source_sentence=sentence.text,
                score=score,
            ))
            stats["ok"] += 1
            question_idx += 1

        out.sort(key=lambda x: (-x.score, x.question_id))

        # Диагностический вывод
        types_str = " ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        print(
            f"[TRAINER] {chunk_id}: "
            f"sents={stats['total']} short={stats['short']} long={stats['long']} "
            f"no_type={stats['no_type']} no_pattern={stats['no_pattern']} "
            f"invalid={stats['invalid']} low_score={stats['low_score']} "
            f"generated={stats['ok']}  types=[{types_str}]"
        )
        return out

    # ── Покрытие ключевых слов ────────────────────────────────────────────────

    def _keyword_coverage(self, sentence: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        sentence = f" {sentence} "
        matches = 0
        for keyword in keywords:
            if f" {keyword.lower()} " in sentence:
                matches += 1
        return matches / len(keywords)

    # ── Проверка ответа ───────────────────────────────────────────────────────

    def check_answer(self, answer: str, question: Question) -> Tuple[bool, float]:
        """Проверяем ответ пользователя. Возвращает (correct, score)."""
        if not answer:
            return False, 0.0
        if len(answer.split()) < 2:
            return False, 0.0

        answer = clean_text_for_check(answer)

        cov = self._keyword_coverage(answer, question.keywords)

        # Один батч-вызов вместо трёх отдельных
        scores = self.rerank_model.predict(
            [
                (answer, question.question),
                (answer, question.source_sentence),
                (answer, question.answer),
            ],
            show_progress_bar=False,
        )
        q_score, src_score, gold_score = float(scores[0]), float(scores[1]), float(scores[2])

        final_score = (
            self.q_score_weight    * q_score
            + self.src_score_weight  * src_score
            + self.gold_score_weight * gold_score
            + 0.05 * cov
        )

        return final_score >= self.answer_threshold, final_score
