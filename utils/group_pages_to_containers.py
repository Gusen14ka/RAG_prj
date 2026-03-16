import re
from typing import List, Dict, Any
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s*(.*)$", flags=re.M)

def page_to_text(page_layout) -> str:
    parts = []
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            t = element.get_text().replace("\x0c", "").rstrip()
            if t:
                parts.append(t)
    return "\n".join(parts)


def group_pages_to_containers(pdf_path: str) -> List[Dict[str,Any]]:
    """
    Возвращает список контейнеров-пунктов:
    { 'section_id': '1.1.2' or 'preface', 'pages': [1,2], 'page_texts':[...], 'first_lines': '...' }
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
        if m:
            sec = m.group(1).strip()
            if current is not None:
                if current["section_id"] == sec:
                    current["pages"].append(page_no)
                    current["page_texts"].append(ptext)
                else:
                    containers.append(current)
                    current = {"section_id": sec, "pages":[page_no], "page_texts":[ptext], "first_lines": first_lines}
            else:
                current = {"section_id": sec, "pages":[page_no], "page_texts":[ptext], "first_lines": first_lines}
        else:
            if current is None:
                # Контейнер предисловия
                current = {"section_id":"preface", "pages":[page_no], "page_texts":[ptext], "first_lines": first_lines}
            else:
                current["pages"].append(page_no)
                current["page_texts"].append(ptext)
    if current is not None:
        containers.append(current)
    return containers