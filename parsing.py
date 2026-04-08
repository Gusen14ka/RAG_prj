import re
import json
from typing import List, Tuple, Dict, Any
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from utils.recursive_chunking import recursive_chunking


CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
HEADER_RE = re.compile(  # ИСПРАВЛЕНО: убран лишний псевдоним `pattern`
    r"^\s*"                    # Пробелы в начале
    r"(?:(\d+(?:\.\d+)*)\.?)?"  # ГРУППА 1: опциональный номер (1 или 1.2.3)
    r"\s*"                      # Пробелы между номером и текстом
    r"(.+?)"                    # ГРУППА 2: текст
    r"(?:\s*\(([^)]+)\))?"      # ГРУППА 3: опциональные скобки
    r"\s*$", 
    flags=re.M
)


# -----------------------------
# 1. Извлечение текста страницы
# -----------------------------

def page_to_text(page_layout) -> str:
    parts = []
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            t = element.get_text().replace("\x0c", "").rstrip()
            if t:
                parts.append(t)
    return "\n".join(parts)


# -----------------------------
# 3. Очистка текста
# -----------------------------

from typing import Optional  # ИСПРАВЛЕНО: убран дублирующий `import re`

DEFAULT_FORMULA_TOKEN = "<FORMULA>"

def cleaning_text(
    text: str,
    *,
    remove_latin: bool = True,
    remove_digits: bool = False,
    replace_formulas_with_token: bool = True,
    formula_token: str = DEFAULT_FORMULA_TOKEN,
    aggressive_math_removal: bool = False,
) -> str:
    """
    Cleans text extracted from PDF for embedding.
    - remove_latin: удалять латинские буквы (True по умолчанию)
    - remove_digits: удалять все цифры (False по умолчанию)
    - replace_formulas_with_token: заменять найденные формулы на token (True по умолчанию)
    - aggressive_math_removal: при True дополнительно заменяет куски с math-символами на token
    """
    if not text:
        return ""

    # 1) нормализуем переводы строки
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) удаляем некоторые pdf-артефакты
    # (cid:123), page numbers вида "\n62 / 1738" и явные "Page 12" в конце строки
    text = re.sub(r"\(cid:\d+\)", " ", text)
    # номера страниц типа "62 / 1738" (в конце строки или после перевода строки)
    text = re.sub(r"\n+\s*\d+\s*/\s*\d+\s*(?=\n|$)", "\n", text)
    # явные "Page 12" или "Стр. 12" в отдельной строке
    text = re.sub(r"(?m)^\s*(Page|Стр\.?|стр\.?)\s*\d+\s*$", " ", text)

    # 3) склеиваем дефисные переносы: "мно-\nжество" -> "множество"
    text = re.sub(r"([^\W\d_])-\s*\n\s*([^\W\d_])", r"\1\2", text, flags=re.U)
    
    # 3.5) убираем артефакты переносов слов со спробелами: "мно- жество" -> "множество"
    text = re.sub(r"([^\W\d_])-\s{1,2}([^\W\d_])", r"\1\2", text, flags=re.U)

    # 4) объединяем много переносов в максимум 2 (чтобы сохранять абзацы)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5) сначала выявим и заменим блоки формул / выражений (если включено)
    if replace_formulas_with_token:
        # 5.1 LaTeX inline math $...$
        text = re.sub(r"\$.*?\$", f" {formula_token} ", text, flags=re.S)

        # 5.2 \[ ... \] and \( ... \)
        text = re.sub(r"\\\[.*?\\\]", f" {formula_token} ", text, flags=re.S)
        text = re.sub(r"\\\(.*?\\\)", f" {formula_token} ", text, flags=re.S)

        # 5.3 короткие выражения в скобках содержащие много математических символов
        text = re.sub(
            r"\([^\n]{0,200}?[=+\-*/^_<>≡≤≥≈≠∑∏∫∂πσλθ±]\s*[^\n]{0,200}?\)",
            f" {formula_token} ",
            text,
        )

        # 5.4 агрессивно: куски с нескольких подряд math-символов/латиницей+символами
        if aggressive_math_removal:
            text = re.sub(
                r"[A-Za-z0-9_]*[=+\-*/^_<>∑∏∫∂πσλθ±≤≥≈≠][A-Za-z0-9_,\.\s\{\}\[\]\(\)]{0,200}",
                f" {formula_token} ",
                text,
            )

    # 6) удаляем математические символы, которые не попали в формулы
    math_symbols = r"[≡∀∃⊆⊂∪∩Δ∑∏⊕∞≤≥≈≠∂∇πσλθ±∫]"
    text = re.sub(math_symbols, " ", text)

    # 7) убираем скобки и фигурные скобки, угловые — но аккуратно (заменяем на пробел)
    text = re.sub(r"[{}\[\]<>|]", " ", text)

    # 8) удаляем ключевые алгоритмические токены как слова (case-insensitive)
    text = re.sub(r"\b(for|while|yield|return|do|end)\b", " ", text, flags=re.I)

    # 9) безопасно удалить латиницу (если включено)
    if remove_latin:
        text = re.sub(r"[A-Za-z]", " ", text)

    # 10) удалить цифры (опционально)
    if remove_digits:
        text = re.sub(r"\d+", " ", text)

    # 11) убрать номера заголовков в стиле "1. " или "1.1. " в начале строки
    # (многострочный режим)
    text = re.sub(r"(?m)^\s*(\d+(?:\.\d+)*\.)\s*", "", text)

    # 12) сокращаем повторяющиеся точки/запятые/пробелы
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[,\s]{2,}", " ", text)

    # 13) удаляем оставшиеся одиночные длинные пунктуационные кучи (напр. ".,.,.,")
    text = re.sub(r"([.,;:()\-\—]){3,}", r"\1", text)

    # 14) финальная нормализация пробелов и обрезка
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = text.strip()

    return text


def group_pages_to_containers(pdf_path: str) -> List[Dict[str,Any]]:
    """
    Возвращает список контейнеров-пунктов:
    { 'section_id': '1.1.2' or 'preface', 'page_texts':'...', 'metadata': {'section':'', 'subsection':''}}
    """
    containers=[]
    current=None
    page_no = 0
    for page_layout in extract_pages(pdf_path):
        page_no += 1
        ptext = page_to_text(page_layout)
        # Подефолту убираем лишнюю инфу - номер страницы в конце текста
        ptext = re.sub(r'\n+\d+\s*/\s*\d+\s*$', '', ptext)
        # берём первую не нулевую строку
        first_lines = "\n".join([ln for ln in ptext.splitlines() if ln.strip()][:3])
        m = HEADER_RE.match(first_lines) if first_lines else None
        # group(1) опциональна — страница без номера не является заголовком нового раздела
        has_numbered_header = m is not None and m.group(1) is not None
        if has_numbered_header:
            sec = m.group(1).strip() # type: ignore
            if current is not None:
                if current["section_id"] == sec:
                    current["page_texts"].append(ptext)
                else:
                    containers.append(current)
                    current = {"section_id": sec, "page_texts": [ptext],
                               "metadata":{"section": current["metadata"]["section"], "subsection": m.group(2).strip()}} # type: ignore
            else:
                current = {"section_id": sec, "page_texts": [ptext],
                           "metadata":{"section": m.group(2).strip(), "subsection": ""}} # type: ignore
        else:
            if current is None:
                # Контейнер предисловия (страницы до первого нумерованного раздела)
                current = {"section_id":"preface", "page_texts": [ptext],
                           "metadata":{"section": "", "subsection": ""}}
            else:
                current["page_texts"].append(ptext)
    if current is not None:
        containers.append(current)
    return containers

"""
Вид метаданных
metadata = {'section': str, 'subsection': str}
"""
def add_metadata_to_text(text: str, metadata: Dict[str,str]) -> str:
    section = metadata.get("section", "")
    subsection = metadata.get("subsection", "")
    title = ""
    if section != "":
        title = "Раздел: " + section + "\n"
    if subsection != "":
        title += "Подраздел: " + subsection + "\n"
    if title != "":
        text = title + text
    return text


# -----------------------------
# 4. Основной pipeline
# -----------------------------

def pdf_to_chunks(pdf_path):

    chunks = []
    containers = group_pages_to_containers(pdf_path)
    for cont in containers:
        cont_chunk = recursive_chunking(cont)
        for ch in cont_chunk:
            # print(ch["chunk_id"])
            # print(len(ch["text"].split()))
            # print(ch["text"][:20])
            # print(ch["metadata"])
            chunks.append(ch)
        
    
    smart_chunks = []
    for chunk in chunks:
        clean_text = cleaning_text(chunk["text"])
        smart_chunk = {"chunk_id": chunk["chunk_id"], "clean_text": clean_text, "raw_text":chunk["text"],
                      "metadata": chunk["metadata"]}
        # print(smart_chunk["chunk_id"])
        # print(smart_chunk["clean_text"][:50])
        # print("raw text:")
        # print(smart_chunk["raw_text"][:50])
        # print()
        smart_chunk["clean_text"] = add_metadata_to_text(smart_chunk["clean_text"], smart_chunk["metadata"])
        smart_chunk["raw_text"] = add_metadata_to_text(smart_chunk["raw_text"], smart_chunk["metadata"])
        smart_chunks.append(smart_chunk)

    return smart_chunks


# -----------------------------
# 5. Сохранение
# -----------------------------
def save_chunks_with_key(chunks, path):
    """Save chunks to a JSON file where the top-level keys are chunk_id.

    The resulting file will be a single JSON object in the form:
      {"<chunk_id>": { ...chunk... }, ...}

    Raises:
        ValueError: if there are duplicate chunk_id values.
    """

    chunks_by_id = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid in chunks_by_id:
            raise ValueError(f"Duplicate chunk_id: {cid}")
        chunks_by_id[cid] = c

    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks_by_id, f, ensure_ascii=False, indent=2)

def save_chunks(chunks, path, path_with_key):

    with open(path, "w", encoding="utf-8") as f:

        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    save_chunks_with_key(chunks, path_with_key)



# -----------------------------
# 6. Запуск
# -----------------------------

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    pdf_file = "test.pdf"
    output_file = "data/chunks.jsonl"
    output_file_with_key = "data/chunks_with_key.json"

    chunks = pdf_to_chunks(pdf_file)
    # ИСПРАВЛЕНО: save_chunks требует 3 аргумента
    save_chunks(chunks, output_file, output_file_with_key)
    print(f"Создано чанков: {len(chunks)}")