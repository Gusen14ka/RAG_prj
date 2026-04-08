from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
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
INLINE_HEADER_RE = re.compile(r"^(?:\d+\.[\d.]*\s+)?(?:Раздел:|Подраздел:|Примечание:).*?(?=\n(?:Раздел:|Подраздел:|[А-ЯЁ]|\d))", flags=re.MULTILINE | re.DOTALL)

# Черный список — слова, которые указывают на плохие ответы
BLACKLIST_WORDS = {
    "раздел", "подраздел", "примечание", "парграф", "введение", "заключение",
    "содержание", "предисловие", "изложение", "сущность", "основной", "главный"
}

def remove_headers(text: str) -> str:
    """Удаляем строки типа 'Раздел: ...', 'Подраздел: ...', 'Примечание: ...' и нумерацию в начале текста."""
    lines = text.split('\n')
    
    # Первый проход: найти индекс первой "настоящей" строки
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Пропускаем заголовочные строки и номера разделов
        if (HEADER_RE.match(stripped) or
            re.match(r"^(?:№\s*)?\d+(?:\.\d+)*\.?\s+", stripped) or
            stripped == ""):
            start_idx = i + 1
        else:
            break
    
    clean_lines = lines[start_idx:]
    # Если первая оставшаяся строка выглядит как заголовок раздела, убираем её
    if clean_lines:
        first_line = clean_lines[0].strip()
        if (1 < len(first_line.split()) <= 6 and
            not re.search(r"[.!?;:—–-]", first_line) and
            re.match(r"^[А-ЯЁ][А-Яа-яё\s]+$", first_line)):
            clean_lines = clean_lines[1:]

    text = '\n'.join(clean_lines).strip()
    
    # Удаляем внутренние заголовки с формулами и символами, которые попали в текст
    text = re.sub(r'\s+[А-ЯЁ][А-Яа-яё\s-]*[∈∅]\s*—[\s\S]*', '', text)
    
    # Убираем короткие заголовки, которые остались в начале строки без отдельного переноса
    text = re.sub(r'^(?:[А-ЯЁ][а-яё]+(?:\s+[а-яё]+){0,4})\s+(?=[А-ЯЁ][а-яё]*\b)', '', text)
    
    # Убираем номера типа "№ 1.1.1." и номера в начале строк
    text = re.sub(r"№\s*\d+(?:\.\d+)*\.?\s*", "", text)
    text = re.sub(r"(?m)^\s*\d+(?:\.\d+)*\.?\s*", "", text)
    # Убираем номера вроде "1.1.5. " внутри текста перед заголовком
    text = re.sub(r"\b\d+(?:\.\d+)*\.?\s+(?=[А-ЯЁ])", "", text)
    
    # Удаляем стандартные артефакты из презентации
    text = re.sub(r'Название\s+параграфа\s+Ключевые\s+термины\s+и\s+обозначения', '', text)
    text = re.sub(r'Ключевые\s+термины\s+и\s+обозначения', '', text)
    text = re.sub(r'Название\s+параграфа', '', text)
    
    # Удаляем оставшиеся короткие строковые заголовки
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

def normalize_spaces(text: str) -> str:
    """Заменяем множественные пробелы одним."""
    return MULTISPACE_RE.sub(" ", text).strip()

def clean_text(text: str) -> str:
    """Базовая очистка от пробелов и мягких дефисов."""
    text = text.replace("\u00ad", "")
    text = text.lower().strip()
    text = text.replace("ё", "е")
    return normalize_spaces(text)

def get_lemma(token: Any) -> str:
    """Вычленяем лемму слова."""
    lemma = getattr(token, "lemma", None)
    if lemma:
        return str(lemma).lower()
    return str(getattr(token, "text", "")).lower()

def sentence_lemmas(sentence: Any) -> List[str]:
    """Вычленяем леммы из предложения."""
    return [get_lemma(t) for t in sentence.tokens]

def has_any_lemma(lemmas: Sequence[str], candidates: Sequence[str]) -> bool:
    """Проверяем наличие элементов lemmas в candidates."""
    s = set(lemmas)
    return any(c in s for c in candidates)

def find_first_fragment(sentence: str, fragment: str) -> Optional[Tuple[int, int]]:
    """Находим первое вхождение fragment в sentence."""
    if not fragment or len(fragment) < 2:
        return None
    idx = sentence.lower().find(fragment.lower())
    if idx < 0:
        return None
    return idx, idx + len(fragment)

def dedupe_preserve(items: Iterable[str]) -> List[str]:
    """Убираем дубликаты, сохраняя порядок."""
    seen = set()
    out: List[str] = []
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

def is_valid_question(question: str, answer: str) -> bool:
    """Проверяем валидность вопроса и ответа."""
    if not question or not answer:
        return False
    
    # Вопрос должен содержать ?
    if '?' not in question:
        return False
    
    # Вопрос должен быть достаточно длинным
    if len(question.split()) < 3:
        return False
    
    # Ответ не должен содержать многоточие (это артефакт)
    if '...' in answer or '…' in answer:
        return False
    
    # Ответ не должен быть фрагментом заголовка
    if answer.split()[0].lower() in BLACKLIST_WORDS:
        return False
    
    # Ответ должен содержать хотя бы одно кириллическое слово
    if not re.search(r'[а-яё]', answer.lower()):
        return False
    
    return True


@dataclass
class Question:
    question_id: str
    question: str
    answer: str
    keywords: List[str]
    source_chunk_id: str
    source_sentence: str
    score: float

class Trainer:
    def __init__(
            self, 
            rerank_model, 
            question_bank_path: str, 
            answer_threshold:float,
            q_score_weight: float,
            src_score_weight: float,
            gold_score_weight: float) -> None:
        self.question_bank_path = question_bank_path
        self.rerank_model = rerank_model
        self.answer_threshold = answer_threshold
        self.q_score_weight = q_score_weight
        self.src_score_weight = src_score_weight
        self.gold_score_weight = gold_score_weight
        # Инициализируем объекты классов Наташи
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

        # YAKE для извлечения ключевых слов
        self.kw_extractor = yake.KeywordExtractor(
            lan="ru", n=3, top=7, dedupLim=0.9, windowsSize=2
        )

    def save_questions(self, questions: List[Question]) -> None:
        """Сохраняем вопросы в JSONL."""
        with open(self.question_bank_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекаем ключевые слова через YAKE."""
        if len(text.split()) < 3:
            return []
        
        items = self.kw_extractor.extract_keywords(text)
        keywords = []
        for phrase, score in items:
            phrase = phrase.strip().strip(",.;:!?")
            if phrase and len(phrase.split()) <= 3:
                keywords.append(phrase)
        
        return dedupe_preserve(keywords)[:5]

    def _classify_sentence(self, sentence: Any) -> Optional[str]:
        """Классифицируем тип предложения по лексико-синтаксическим маркерам."""
        lemmas = sentence_lemmas(sentence)
        text = sentence.text.lower()

        # Определения (самый надёжный тип)
        if has_any_lemma(lemmas, ["это", "называться", "являться", "представлять", "быть"]):
            return "definition"

        # Состав / включение
        if has_any_lemma(lemmas, ["включать", "состоять", "содержать", "входить", "состав"]):
            return "composition"

        # Свойства / характеристики
        if has_any_lemma(lemmas, ["характеризовать", "характеризоваться", "свойство", "определяться", "иметь"]):
            return "property"

        # Зависимости
        if has_any_lemma(lemmas, ["зависеть", "влиять", "как", "связь"]):
            return "dependency"

        # Сравнение
        if "в отличие" in text or has_any_lemma(lemmas, ["отличаться", "сравнить", "различие", "подобно"]):
            return "comparison"

        # Причина / следствие
        if any(word in text for word in ["потому", "поэтому", "причина", "вследствие", "из-за", "так как", "потому что"]):
            return "cause"

        return None

    def _generate_question(self, sentence: str, kind: str, doc: Doc) -> Tuple[str, str, float]:
        """Генерируем пару вопрос-ответ. Возвращаем также confidence score."""
        s = normalize_spaces(sentence)
        
        # Очищаем от встроенных номеров типа "1.1 Название" которые попали в начало
        s = re.sub(r"^\d+\.\d+(\.\d+)?\s+", "", s).strip()
        
        # Только ЧЕТКИЕ типы вопросов
        if kind == "definition":
            patterns = [
                (r"термин\s+«([^»]+)»\s+для\s+обозначения\s+(.+?)(?:[.;!?]|$)", 0.92),
                (r"^(.+?)\s*(?:—|-)\s*это\s+(.+?)(?:[.;!?]|$)", 0.95),
                (r"^(.+?)\s+это\s+(.+?)(?:[.;!?]|$)", 0.90),
                (r"^(.+?)\s+называется\s+(.+?)(?:[.;!?]|$)", 0.85),
            ]
            
            for pattern, conf in patterns:
                m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    try:
                        subject = m.group(1).strip().strip(",.;:-")
                        answer = m.group(2).strip().strip(",.;:-")
                        
                        # Очищаем subject от встроенных номеров
                        subject = re.sub(r"^\d+\.\d+(\.\d+)?\s+", "", subject).strip()
                        
                        # Не оставляем предлоги в начале темы
                        if subject.split()[0].lower() in {"при", "в", "на", "по", "для", "из", "с", "через", "со", "от"}:
                            continue
                        
                        # Валидация
                        if not subject or len(subject) < 3 or len(answer) < 3:
                            continue
                        if len(answer.split()) > 12:
                            continue
                        if not is_valid_question(f"Что такое {subject}?", answer):
                            continue
                        
                        question = f"Что такое {subject}?"
                        return question, answer, conf
                    except:
                        pass
        
        elif kind == "composition":
            patterns = [
                (r"^(.+?)\s+состоит\s+из\s+(.+?)(?:[.;!?]|$)", 0.92),
                (r"^(.+?)\s+включает\s+(.+?)(?:[.;!?]|$)", 0.88),
            ]
            
            for pattern, conf in patterns:
                m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    try:
                        subject = m.group(1).strip().strip(",.;:-")
                        answer = m.group(2).strip().strip(",.;:-")
                        
                        # Очищаем subject от встроенных номеров
                        subject = re.sub(r"^\d+\.\d+(\.\d+)?\s+", "", subject).strip()
                        
                        if not subject or len(subject) < 3 or len(answer) < 3:
                            continue
                        if len(answer.split()) > 12:
                            continue
                        
                        question = f"Из чего состоит {subject}?" if "состоит" in s.lower() else f"Что включает {subject}?"
                        if not is_valid_question(question, answer):
                            continue
                        
                        return question, answer, conf
                    except:
                        pass
        
        elif kind == "dependency":
            patterns = [
                (r"^(.+?)\s+зависит\s+от\s+(.+?)(?:[.;!?]|$)", 0.85),
                (r"^(.+?)\s+влияет\s+на\s+(.+?)(?:[.;!?]|$)", 0.83),
            ]
            
            for pattern, conf in patterns:
                m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    try:
                        subject = m.group(1).strip().strip(",.;:-")
                        answer = m.group(2).strip().strip(",.;:-")
                        
                        # Очищаем subject от встроенных номеров
                        subject = re.sub(r"^\d+\.\d+(\.\d+)?\s+", "", subject).strip()
                        
                        if not subject or len(subject) < 3 or len(answer) < 3:
                            continue
                        if len(answer.split()) > 10:
                            continue
                        
                        question = f"От чего зависит {subject}?" if "зависит" in s.lower() else f"На что влияет {subject}?"
                        if not is_valid_question(question, answer):
                            continue
                        
                        return question, answer, conf
                    except:
                        pass
        
        # НЕ используем fallback! Если тип не распознан или не подошёл — не генерируем вопрос
        return "", "", 0.0

    def _score_item(
        self, qtype: str, sentence: str, question: str, answer: str, keywords: Sequence[str], conf: float
    ) -> float:
        """Оцениваем качество пары вопрос-ответ."""
        # Базовая оценка по типу * confidence от регулярного выражения
        base_scores = {
            "definition": 1.0,
            "composition": 0.9,
            "property": 0.85,
            "dependency": 0.80,
        }
        base = base_scores.get(qtype, 0.0) * conf

        penalty = 0.0
        
        # Очень длинные предложения = менее надежны
        sent_len = len(sentence.split())
        if sent_len > 60:
            penalty += 0.40
        elif sent_len > 40:
            penalty += 0.20
        
        # Очень длинные ответы
        ans_len = len(answer.split())
        if ans_len > 15:
            penalty += 0.25
        elif ans_len > 10:
            penalty += 0.10
        
        # Если нет ключевых слов
        if not keywords:
            penalty += 0.20

        bonus = 0.0
        
        # Ответ явно присутствует в предложении
        if answer.lower() in sentence.lower():
            bonus += 0.15
        
        # Вопрос выглядит естественно
        if question.endswith("?") and len(question.split()) >= 3:
            bonus += 0.10

        score = max(0.0, min(1.0, base + bonus - penalty))
        return round(score, 4)

    def build_question_bank(self, text: str, chunk_id: str, min_score: float = 0.65) -> List[Question]:
        """Генерируем вопросники из текста."""
        # Удаляем заголовки раздела
        text = remove_headers(text)
        
        # Удаляем строковые заголовки, если остались
        first_line = text.split('\n')[0].strip() if text else ""
        if first_line and re.match(r"^[А-ЯЁ][А-Яа-яЁё\s]+$", first_line) and len(first_line.split()) <= 5:
            text = '\n'.join(text.split('\n')[1:]).strip()

        # Удаляем номера в начале строк, которые могли остаться после очистки
        text = re.sub(r'(?m)^\s*\d+(?:\.\d+)*\.?\s*', '', text)
        text = re.sub(r'№\s*\d+(?:\.\d+)*\.?\s*', '', text)
        
        text = clean_text(text)
        
        out: List[Question] = []
        question_idx = 1

        doc = Doc(text)
        doc.segment(self.segmenter)

        if not doc.sents or not doc.tokens:
            return out

        # Парсим морфологию, синтаксис, NER
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)
        
        if doc.spans:
            for span in doc.spans:
                try:
                    span.normalize(self.morph_vocab)
                except:
                    pass

        # Обрабатываем каждое предложение
        for sentence in doc.sents:
            sent_words = len(sentence.text.split())
            
            # Слишком короткие предложения пропускаем
            if sent_words < 6:
                continue
            
            # Слишком длинные тоже не полезны
            if sent_words > 80:
                continue

            qtype = self._classify_sentence(sentence)
            if qtype is None:
                continue

            question, answer, regex_conf = self._generate_question(sentence.text, qtype, doc)
            
            # Если регулярное выражение не подошло
            if not question or not answer or regex_conf == 0.0:
                continue
            
            # Двойная проверка валидности
            if not is_valid_question(question, answer):
                continue

            keywords = self._extract_keywords(sentence.text)
            score = self._score_item(qtype, sentence.text, question, answer, keywords, regex_conf)

            # ЖЕСТКИЙ фильтр: только хорошие вопросы
            if score < min_score:
                continue

            out.append(Question(
                question_id=f"{chunk_id}__q_{question_idx:05d}",
                question=question,
                answer=answer,
                keywords=keywords,
                source_chunk_id=chunk_id,
                source_sentence=sentence.text,
                score=score
            ))
            question_idx += 1

        # Сортируем по оценке
        out.sort(key=lambda x: (-x.score, x.question_id))
        return out
    
    def _keyword_coverage(self, sentence: str, keywords: List[str]) -> float:
        """Вычисляем вхождение keyword в предложение"""
        if not keywords:
            return 1.0
        sentence = f" {sentence} "
        mathes = 0

        for keyword in keywords:
            kw_norm = keyword.lower()
            kw_norm = f" {kw_norm} "
            if kw_norm and kw_norm in sentence:
                mathes += 1

        return mathes / len(keywords)
    
    def check_answer(self, answer: str, question: Question) -> Tuple[bool, float]:
        """Проверяем ответ пользователя. Возвращает [правильно/неправильно, score, source_sentence]"""
        if not answer:
            return False, 0.0
        
        if len(answer.split()) < 2:
            return False, 0.0
        answer = clean_text(answer)

        # Скорее всего лучше как-то проверять с помощью наташи или вообще не учитывать
        cov = self._keyword_coverage(answer, question.keywords)

        # Проводим сравнение рерак-моделью:
        q_score = self.rerank_model.predict([(answer, question.question)], show_progress_bar=False)[0]
        src_score = self.rerank_model.predict([(answer, question.source_sentence)], show_progress_bar=False)[0]
        gold_score = self.rerank_model.predict([(answer, question.answer)], show_progress_bar=False)[0]

        final_score = (self.q_score_weight * q_score 
                       + self.src_score_weight * src_score 
                       + self.gold_score_weight * gold_score 
                       + 0.05 * cov)
        
        return final_score >= self.answer_threshold, final_score










