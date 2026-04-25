import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LTTextContainer
from utils.recursive_chunking import recursive_chunking
#from parsing_llm import make_clean_for_embedding_llm


# ---------------------------------------------------------------------------
# Паттерн нумерованного заголовка — применяем ТОЛЬКО к первой строке страницы
# Примеры: "1.", "1.1.", "1.1.3.", "1.2.3.4"
# ---------------------------------------------------------------------------
SECTION_ID_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*\.?)\s+(.+?)(?:\s*\(\d+/\d+\))?\s*$"
)

# Признаки строки-заголовка таблицы терминов
_TABLE_HEADER_RE = re.compile(
    r"Название параграфа|Назв\.\s*параграфа|Ключевые термины|№"
)

# Паттерн строки из одних только номеров параграфов: "1.2.3." или "1.2.3. *"
_SECTION_NUM_ONLY_RE = re.compile(
    r"^\s*\d+\.\d+\.?\d*\.?\s*\*?\s*$"
)


# ---------------------------------------------------------------------------
# 1. Вспомогательные
# ---------------------------------------------------------------------------

def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[а-яёА-ЯЁ]", text))


def _is_table_element(text: str) -> bool:
    """
    Возвращает True если текстовый блок выглядит как часть таблицы терминов.

    Признаки:
    - содержит явный заголовок таблицы ("Название параграфа", "№" и т.п.)
    - ИЛИ является строкой из одного только номера параграфа ("1.2.3.")
    """
    stripped = text.strip()
    if _TABLE_HEADER_RE.search(stripped):
        return True
    if _SECTION_NUM_ONLY_RE.match(stripped):
        return True
    return False


# ---------------------------------------------------------------------------
# 2. Извлечение текста страницы с фильтрацией таблиц
# ---------------------------------------------------------------------------

def page_to_text(page_layout: LTPage, filter_tables: bool = True) -> Tuple[str, bool]:
    """
    Извлекает текст страницы.
    Возвращает (text, had_table) где had_table=True если на странице была таблица.

    Логика фильтрации:
    - Как только встречаем элемент с признаком таблицы — все последующие
      элементы до конца страницы отбрасываем (таблица идёт до конца страницы).
    - Элементы ДО таблицы (обычный текст введения раздела) сохраняем.
    - Это корректно обрабатывает случай когда таблица занимает нижнюю часть
      страницы (напр. стр. 92: сверху вводный текст, снизу таблица).
    """
    parts = []
    had_table = False

    for element in page_layout:
        if not isinstance(element, LTTextContainer):
            continue
        t = element.get_text().replace("\x0c", "").rstrip()
        if not t.strip():
            continue

        if filter_tables and _is_table_element(t):
            had_table = True
            break  # всё что идёт после — тоже таблица, пропускаем

        parts.append(t)

    text = "\n".join(parts)
    return text, had_table


# ---------------------------------------------------------------------------
# 3. section_id из первой строки страницы
# ---------------------------------------------------------------------------

def extract_section_id_from_page(page_layout: LTPage) -> Optional[Tuple[str, str]]:
    """
    Возвращает (section_id, title) если первая строка страницы — нумерованный
    заголовок, иначе None.

    Особый случай: иногда первый элемент — обрывок формулы с предыдущей
    страницы (нет кириллицы). Пропускаем ровно один такой элемент.
    """
    skipped_one = False

    for element in page_layout:
        if not isinstance(element, LTTextContainer):
            continue

        raw = element.get_text().replace("\x0c", "")
        first_line = next(
            (l.strip() for l in raw.splitlines() if l.strip()), ""
        )
        if not first_line:
            continue

        # Мусорный обрывок формулы — пропускаем один раз
        if not _has_cyrillic(first_line) and not skipped_one:
            skipped_one = True
            continue

        m = SECTION_ID_RE.match(first_line)
        if m:
            sec_id = m.group(1).rstrip(".")
            title = m.group(2).strip()
            return sec_id, title

        return None  # первый кириллический элемент не заголовок

    return None


# ---------------------------------------------------------------------------
# 4. Очистка текста
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Названия пунктов/страниц вида "1.2.4. Название (3/5)" в теле чанка
    # (появляются из-за перекрытия чанков и склейки страниц)
    text = re.sub(r"\d+\.(?:\d+\.)+\s+[А-ЯЁ][^\n(]{2,60}\(\d+/\d+\)", "", text)

    # PDF-артефакты (cid:123)
    text = re.sub(r"\(cid:\d+\)", " ", text)

    # Номера страниц
    text = re.sub(r"(?m)^\s*\d+\s*/\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*(Page|Стр\.?|стр\.?)\s*\d+\s*$", "", text, flags=re.I)

    # Дефисные переносы через \n: "обо-\nзначается" → "обозначается"
    text = re.sub(r"([а-яёА-ЯЁa-zA-Z])-[ \t]*\n[ \t]*([а-яёА-ЯЁa-zA-Z])", r"\1\2", text)

    # Дефисные переносы через пробел: "обо- значается" → "обозначается"
    # Только между буквами, только если перед дефисом ≤4 символа от начала слова
    # Признак типографского переноса: строчная + дефис + пробел + строчная
    text = re.sub(r"([а-яё])-[ \t]{1,2}([а-яё])", r"\1\2", text)

    # LaTeX inline: $...$ и $$...$$
    text = re.sub(r"\$\$.*?\$\$", " <FORMULA> ", text, flags=re.S)
    text = re.sub(r"\$[^\$\n]{1,300}?\$", " <FORMULA> ", text, flags=re.S)

    # LaTeX display: \[...\]  \(...\)
    text = re.sub(r"\\\[.*?\\\]", " <FORMULA> ", text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)", " <FORMULA> ", text, flags=re.S)

    # LaTeX-команды: \frac{}{} и т.п.
    text = re.sub(r"\\[a-zA-Z]+\s*(?:\{[^}]{0,100}\})*", " <FORMULA> ", text)

    # Управляющие символы (кроме \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Строки из одинаковых спецсимволов
    text = re.sub(r"(?m)^[\s\W]{0,2}([^\w\n])\1{2,}[\s\W]{0,2}$", "", text)

    # Нормализация пробелов
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?m)^ ", "", text)

    # Не более двух переносов строк
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Одиночные символы-мусор на отдельной строке
    text = re.sub(r"(?m)^[^\w\n]{1,3}$", "", text)

    return text.strip()

# ---------------------------------------------------------------------------
# 4b. make_clean_for_embedding — поверх normalize_text, убирает блоки
#     алгоритмов → clean_text
# ---------------------------------------------------------------------------

# Паттерн начала блока алгоритма: строка начинается с while/for без кириллицы
_ALGO_BLOCK_START_RE = re.compile(
    r"^\s*(while\b|for\b.+\bdo\b|if\b.+\bthen\b)",
    re.I
)
# Паттерн конца блока: end while / end for / end if
_ALGO_BLOCK_END_RE = re.compile(
    r"\b(end\s+while|end\s+for|end\s+if)\b",
    re.I
)
# Инлайн-алгоритм на одной строке: содержит do...end или yield внутри {}
_ALGO_INLINE_RE = re.compile(
    r"while\b[^.]*?\bdo\b.+?\bend\s+while\b"
    r"|for\b[^.]*?\bdo\b.+?\bend\s+for\b"
    r"|if\b[^.]*?\bthen\b.+?\bend\s+if\b",
    re.S | re.I,
)


def _remove_algo_blocks(text: str) -> str:
    """
    Убирает многострочные блоки псевдокода:
        while ... do
            ...
        end while
    Инлайн-алгоритмы внутри {...} не трогает — они часть определений.
    """
    lines = text.splitlines()
    result = []
    in_block = False

    for line in lines:
        stripped = line.strip()

        if in_block:
            # Ищем конец блока — строку с end тоже пропускаем
            if _ALGO_BLOCK_END_RE.search(stripped):
                in_block = False
            continue

        # Начало блока: строка без кириллицы, начинается с while/for/if
        if (not _has_cyrillic(stripped)
                and _ALGO_BLOCK_START_RE.match(stripped)):
            # Если блок закрывается на той же строке — однострочный, оставляем
            if _ALGO_BLOCK_END_RE.search(stripped):
                result.append(line)
            else:
                in_block = True
            continue

        # Одиночный end-маркер без открывающего блока (orphan) — пропускаем
        if (not _has_cyrillic(stripped)
                and _ALGO_BLOCK_END_RE.match(stripped.strip())):
            continue

        result.append(line)

    return "\n".join(result)


def make_clean_for_embedding(text: str) -> str:
    """
    Дополнительная очистка поверх normalize_text для embedding.
    Убирает только то что реально мешает модели и не несёт текстового смысла:
    - многострочные блоки псевдокода (while/for/if ... end)
    - LaTeX-разметку ($...$ и команды \frac{} и т.п.) — заменяем <FORMULA>
    Математические символы (∈, ⊆, α, β...) и инлайн-формулы оставляем.
    """
    if not text:
        return ""

    # LaTeX inline: $...$ и $$...$$
    text = re.sub(r"\$\$.*?\$\$", " <FORMULA> ", text, flags=re.S)
    text = re.sub(r"\$[^\$\n]{1,300}?\$", " <FORMULA> ", text, flags=re.S)

    # LaTeX display: \[...\]  \(...\)
    text = re.sub(r"\\\[.*?\\\]", " <FORMULA> ", text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)", " <FORMULA> ", text, flags=re.S)

    # LaTeX-команды: \frac{}{}, \sum_{} и т.п.
    text = re.sub(r"\\[a-zA-Z]+\s*(?:\{[^}]{0,100}\})*", " <FORMULA> ", text)

    # Повторяющиеся <FORMULA> → один
    text = re.sub(r"(\s*<FORMULA>\s*){2,}", " <FORMULA> ", text)

    # Блоки псевдокода — инлайн и многострочные
    text = _ALGO_INLINE_RE.sub(" <ALGORITHM> ", text)
    text = _remove_algo_blocks(text)

    # Повторяющиеся <ALGORITHM> → один
    text = re.sub(r"(\s*<ALGORITHM>\s*){2,}", " <ALGORITHM> ", text)

    # Висящий открывающий оператор (while/for/if без закрывающего):
    # если после него до конца чанка < 80 символов — убираем хвост
    def _trim_orphan_open(t: str) -> str:
        m = re.search(
            r"(\bwhile\b[^.]*?\bdo\b|\bfor\b[^.]*?\bdo\b|\bif\b[^.]*?\bthen\b)(?!.*\bend\s+(?:while|for|if)\b)",
            t, re.S | re.I
        )
        if m and len(t) - m.start() < 80:
            return t[:m.start()].rstrip()
        return t

    # Висящий закрывающий оператор (end while/for/if без открывающего):
    # срезаем префикс от начала до конца последнего end-маркера,
    # если этот префикс не содержит кириллических слов (значит это чистый код)
    def _trim_orphan_close(t: str) -> str:
        # Ищем последний end-маркер в первых 300 символах тела
        for m in re.finditer(r"\bend\s+(?:while|for|if)\b", t, re.I):
            if m.end() > 300:
                break
            prefix = t[:m.start()]
            # Если до end нет кириллических слов вне комментариев — это хвост алгоритма
            # Убираем комментарии // перед проверкой на кириллицу
            prefix_no_comments = re.sub(r"//[^\n]*", "", prefix)
            if not re.search(r"[а-яёА-ЯЁ]{4,}", prefix_no_comments):
                t = t[m.end():].lstrip()
                # Рекурсивно — может быть несколько подряд (end for end for)
                return _trim_orphan_close(t)
        return t

    text = _trim_orphan_open(text)
    text = _trim_orphan_close(text)


    # Нормализация пробелов после удалений
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ---------------------------------------------------------------------------
# 5. Группировка страниц в контейнеры
# ---------------------------------------------------------------------------

def group_pages_to_containers(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Группирует страницы в контейнеры по нумерованным разделам.
    Таблицы терминов фильтруются: блок таблицы отбрасывается, текст до неё
    (если есть) сохраняется.

    Структура контейнера:
        {
            "section_id": "1.1.3",
            "page_texts": [...],
            "metadata": {"section": str, "subsection": str},
        }
    """
    containers: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for page_layout in extract_pages(pdf_path):
        page_text, _had_table = page_to_text(page_layout, filter_tables=True)
        # Убираем номер страницы в конце
        page_text = re.sub(r"\n+\d+\s*/\s*\d+\s*$", "", page_text)

        header = extract_section_id_from_page(page_layout)

        if header is not None:
            sec_id, title = header

            if current is not None:
                if current["section_id"] == sec_id:
                    # Продолжение того же раздела (напр. "(1/2)" и "(2/2)")
                    if page_text.strip():
                        current["page_texts"].append(page_text)
                    continue
                containers.append(current)

            depth = sec_id.count(".") + 1
            if depth == 1 or depth == 2:
                section_name = title
                subsection_name = ""
            else:
                parent_id = sec_id.rsplit(".", 1)[0]
                parent = next(
                    (c for c in reversed(containers) if c["section_id"] == parent_id),
                    None,
                )
                section_name = parent["metadata"]["section"] if parent else title
                subsection_name = title

            current = {
                "section_id": sec_id,
                "page_texts": [page_text] if page_text.strip() else [],
                "metadata": {
                    "section": section_name,
                    "subsection": subsection_name,
                },
            }
        else:
            if current is None:
                current = {
                    "section_id": "preface",
                    "page_texts": [],
                    "metadata": {"section": "", "subsection": ""},
                }
            if page_text.strip():
                current["page_texts"].append(page_text)

    if current is not None:
        containers.append(current)

    return containers


# ---------------------------------------------------------------------------
# 6. Метаданные в тексте
# ---------------------------------------------------------------------------

def add_metadata_to_text(text: str, metadata: Dict[str, str]) -> str:
    section = metadata.get("section", "")
    subsection = metadata.get("subsection", "")
    title = ""
    if section:
        title = f"Раздел: {section}\n"
    if subsection:
        title += f"Подраздел: {subsection}\n"
    if title:
        text = title + text
    return text


# ---------------------------------------------------------------------------
# 7. Основной pipeline
# ---------------------------------------------------------------------------

def pdf_to_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    chunks = []
    containers = group_pages_to_containers(pdf_path)
    for cont in containers:
        for ch in recursive_chunking(cont):
            chunks.append(ch)

    smart_chunks = []
    count = 0
    for chunk in chunks:
        raw_normalized = normalize_text(chunk["text"])
        if not raw_normalized.strip():
            continue
        
        # count += 1
        
        # algo_flags = ["while", "for", "return", "yield", "end"]
        # if any(flag in raw_normalized for flag in algo_flags):
        #     cleaned = make_clean_for_embedding_llm(raw_normalized)
        # else:
        #     cleaned = raw_normalized
        
        # print("chunk %d / %d", count, len(chunks))

        cleaned = make_clean_for_embedding(raw_normalized)

        meta = chunk["metadata"]
        smart_chunk = {
            "chunk_id":   chunk["chunk_id"],
            "clean_text": add_metadata_to_text(cleaned, meta),
            "raw_text":   add_metadata_to_text(raw_normalized, meta),
            "metadata":   meta,
        }
        smart_chunks.append(smart_chunk)
    return smart_chunks

def pdf_to_plain_text(pdf_path) -> str:
    text = ""
    containers = group_pages_to_containers(pdf_path)
    for cont in containers:
        for ch in recursive_chunking(cont, overlap_words=0):
            normalized = normalize_text(ch["text"])
            if not normalized.strip():
                continue
            text += normalized + "\n"

    return text

def save_plain_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_plain_text(path: str) -> str:
    out = ""
    with open(path, "r", encoding="utf-8") as f:
        out = f.read()
    return out


# ---------------------------------------------------------------------------
# 8. Сохранение
# ---------------------------------------------------------------------------

def save_chunks_with_key(chunks: List[Dict], path: str) -> None:
    chunks_by_id: Dict[str, Any] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid in chunks_by_id:
            raise ValueError(f"Duplicate chunk_id: {cid}")
        chunks_by_id[cid] = c # type: ignore
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks_by_id, f, ensure_ascii=False, indent=2)


def save_chunks(chunks: List[Dict], path: str, path_with_key: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    save_chunks_with_key(chunks, path_with_key)


# ---------------------------------------------------------------------------
# 9. Запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    pdf_file = "doc.pdf"
    output_file = "data/chunks.jsonl"
    output_file_with_key = "data/chunks_with_key.json"

    chunks = pdf_to_chunks(pdf_file)
    save_chunks(chunks, output_file, output_file_with_key)
    print(f"Создано чанков: {len(chunks)}")