"""
extract_terms.py
================
Извлекает термины из таблиц учебника.

Структура таблицы: № | Название параграфа | Ключевые термины и обозначения

Алгоритм:
    1. Находим строки таблицы по номерам в col1 (паттерн «1.4.1.» или «1.4.1. *»).
    2. По координатам этих строк строим интервалы таблицы.
    3. Все строки col3 распределяем по этим интервалам.
    4. Внутри строки склеиваем текст, убираем дефисные переносы, режем по «;».

Зависимости: pip install pymupdf
Использование: python extract_terms.py textbook.pdf
"""

import re
import sys
from typing import List, Tuple

import fitz  # PyMuPDF


ROW_LABEL_RE = re.compile(r"^\d+(?:\.\d+)+\.?\s*\*?$")
PAGE_FOOTER_RE = re.compile(r"^\d+\s*/\s*\d+$")
SECTION_ID_RE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s+(.+?)(?:\s*\(\d+/\d+\))?\s*$")

def _clean_text(text: str) -> str:
    """Убирает артефакты дефисных переносов и лишние пробелы."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u200b", "")  # zero-width space

    # Дефисный перенос через \n: "обо-\nзначается" → "обозначается"
    text = re.sub(
        r"([А-ЯЁа-яёA-Za-z])-\s*\n\s*([А-ЯЁа-яёA-Za-z])",
        r"\1\2",
        text,
    )

    # Дефисный перенос через пробел: "обо- значается" → "обозначается"
    text = re.sub(
        r"([А-ЯЁа-яё])-\s+([А-ЯЁа-яё])",
        r"\1\2",
        text,
    )

    # Множественные пробелы/табы → один пробел
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _get_lines(page: fitz.Page) -> List[Tuple[float, float, float, str]]:
    """
    Возвращает строки страницы:
    (y0, y1, x0, text)
    """
    lines: List[Tuple[float, float, float, str]] = []

    for block in page.get_text("dict")["blocks"]: # type: ignore
        if block.get("type") != 0: # type: ignore
            continue

        for line in block["lines"]: # type: ignore
            spans = line.get("spans", []) # type: ignore
            if not spans:
                continue

            x0 = min(span["bbox"][0] for span in spans)
            y0 = min(span["bbox"][1] for span in spans)
            y1 = max(span["bbox"][3] for span in spans)
            text = "".join(span["text"] for span in spans).strip()

            if text:
                lines.append((y0, y1, x0, text))

    lines.sort(key=lambda item: (item[0], item[2]))
    return lines


def _find_col3_x(page: fitz.Page, fallback: float = 183.0) -> float:
    """Граница 3-го столбца — чуть правее второй вертикальной линии таблицы."""
    xs = sorted(
        set(
            round(d["rect"].x0, 0)
            for d in page.get_drawings()
            if abs(d["rect"].x1 - d["rect"].x0) <= 2
            and abs(d["rect"].y1 - d["rect"].y0) > 5
        )
    )
    return xs[1] + 3.0 if len(xs) >= 2 else fallback


def _build_row_intervals(lines: List[Tuple[float, float, float, str]]) -> List[Tuple[float, float, str]]:
    """
    Строит интервалы строк таблицы по строкам первого столбца.
    Учитывает номера вида:
        1.4.1.
        1.4.1. *
        1.9.2. *
    Возвращает [(y_top, y_bottom, row_id), ...].
    row_id — номер параграфа без точки и звёздочки: «1.1.2. *» → «1.1.2».
    """
    row_ys : List[Tuple[str, str]] = sorted(
        {
            (round(y0, 2), re.match(r"(\d+(?:\.\d+)+)", text).group(1)) # type: ignore
            for y0, y1, x0, text in lines
            if x0 <= 80 and ROW_LABEL_RE.fullmatch(text.strip())
        },
        key=lambda r: r[0]
    )

    intervals: List[Tuple[float, float, str]] = []
    for i, (top, row_id) in enumerate(row_ys):
        bottom = row_ys[i + 1][0] if i + 1 < len(row_ys) else float("inf")
        intervals.append((top, bottom, row_id)) # type: ignore

    return intervals


def extract_terms(pdf_path: str) -> List[Tuple[str, str]]:
    doc = fitz.open(pdf_path)
    all_terms: List[Tuple[str, str]] = []

    for page in doc:
        page_text = page.get_text("text")
        if "Ключевые термины" not in page_text:
            continue

        lines = _get_lines(page)
        intervals = _build_row_intervals(lines)
        if not intervals:
            continue

        col3_x = _find_col3_x(page)
        row_texts: List[List[Tuple[float, float, str]]] = [[] for _ in intervals]

        # Распределяем строки col3 по интервалам строк таблицы
        for y0, y1, x0, text in lines:
            if x0 < col3_x:
                continue

            s = text.strip()

            if "Ключевые термины" in s:
                continue
            if s in {"№", "Название параграфа"}:
                continue
            if PAGE_FOOTER_RE.fullmatch(s):
                continue

            # Берём строку по максимальному пересечению по высоте.
            chosen_idx = None
            best_overlap = 0.0

            for i, (top, bottom, _) in enumerate(intervals):
                overlap = max(0.0, min(y1, bottom) - max(y0, top))
                if overlap > best_overlap:
                    best_overlap = overlap
                    chosen_idx = i

            # Если пересечение нулевое, fallback по центру строки
            if chosen_idx is None or best_overlap <= 0:
                yc = (y0 + y1) / 2.0
                for i, (top, bottom, _) in enumerate(intervals):
                    if top <= yc < bottom:
                        chosen_idx = i
                        break

            if chosen_idx is not None:
                row_texts[chosen_idx].append((y0, x0, text))

        # Внутри каждой строки: склеиваем куски, чистим переносы, делим по ';'
        for chunks, (_, _, row_id) in zip(row_texts, intervals):
            if not chunks:
                continue

            chunks.sort(key=lambda item: (item[0], item[1]))

            raw = "\n".join(chunk_text for _, _, chunk_text in chunks)
            cleaned = _clean_text(raw)
            cleaned = cleaned.replace("\n", " ")

            for term in re.split(r"\s*[;,]\s*", cleaned):
                term = term.strip()
                if term:
                    all_terms.append((term, row_id))

    doc.close()
    return all_terms


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "doc.pdf"
    terms = extract_terms(path)

    print(f"\nНайдено терминов: {len(terms)}\n")
    for i, (t, id) in enumerate(terms, 1):
        print(f"{i:4d}. [{id}] {t}")