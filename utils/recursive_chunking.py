from typing import Optional

from entities import (
    PageContrainer,
    ChunkMetaData,
    Chunk
)

def recursive_chunking(
    section_obj: PageContrainer,
    chunk_size_words: int = 200,
    overlap_words: int = 20,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    section_id = section_obj.section_id
    page_texts = section_obj.page_texts
    metadata = section_obj.metadata
    if separators is None:
        # порядок сепараторов: сначала крупные (абзацы), затем мелкие (слова)
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def words_count(s: str) -> int:
        return len(s.split())

    # объединяем страницы, сохраняя границу страницы как двойной перевод строки - ПОКА ЧТО КАК ОДИН \n
    full_text = ("\n").join(p.strip() for p in page_texts if p and p.strip())

    # рекурсивно разбивает текст на сегменты, не превышающие chunk_size_words,
    # пытаясь использовать каждый сепаратор по очереди
    def split_recursive(text: str, sep_idx: int = 0) -> list[str]:
        text = text.strip()
        if not text:
            return []
        # если уже достаточно коротко — вернуть как сегмент
        if words_count(text) <= chunk_size_words:
            return [text]
        # если вышли за пределы списка сепараторов — жёсткий split по словам
        if sep_idx >= len(separators):
            words = text.split()
            segments = []
            # делаем жёсткие куски по chunk_size_words
            for i in range(0, len(words), chunk_size_words):
                seg = " ".join(words[i : i + chunk_size_words])
                segments.append(seg)
            return segments

        sep = separators[sep_idx]
        # если сепаратор пустой (не ожидается, но на всякий) — перейти дальше
        if sep == "":
            return split_recursive(text, sep_idx + 1)

        # split по текущему сепаратор (preserve separator inside pieces? нет — убираем его)
        parts = [p.strip() for p in text.split(sep) if p and p.strip()]
        # если split не разделил (т.е. нет sep в тексте) — перейти к следующему сепаратор
        if len(parts) == 1:
            return split_recursive(text, sep_idx + 1)

        result: list[str] = []
        for part in parts:
            # рекурсивно обрабатываем каждую часть
            result.extend(split_recursive(part, sep_idx + 1))
        return result

    # получаем набор семантических сегментов, каждый <= chunk_size_words (или результат жёсткого сплита)
    segments = split_recursive(full_text, 0)

    # теперь собираем финальные чанки: аккуратно группируем сегменты в чанки до chunk_size_words,
    # и применяем overlap (в словах)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_words = 0

    def make_chunk_from_parts(parts: list[str]) -> str:
        return " ".join(p.strip() for p in parts if p and p.strip())

    i = 0
    while i < len(segments):
        seg = segments[i]
        seg_wc = words_count(seg)
        # если один сегмент сам больше, чем chunk_size_words (редко — после жёсткого сплита) —
        # тогда вынем его как отдельный chunk (всё равно не превышает, т.к. жёсткий сплит уже сделал куски)
        if seg_wc >= chunk_size_words and current_words == 0:
            chunk_text = seg.strip()
            chunks.append(Chunk(f"{section_id}:{len(chunks)+1}", chunk_text, chunk_text, metadata))
            i += 1
            continue

        # если добавление сегмента не превышает размер — добавляем в текущую сборку
        if current_words + seg_wc <= chunk_size_words or current_words == 0:
            current_parts.append(seg)
            current_words += seg_wc
            i += 1
            # если ровно достигли — создаём chunk
            if current_words >= chunk_size_words:
                chunk_text = make_chunk_from_parts(current_parts)
                chunks.append(Chunk(f"{section_id}:{len(chunks)+1}", chunk_text, chunk_text, metadata))
                # подготовка к следующему чанку: оставляем overlap_words последних слов
                last_words = chunk_text.split()[-overlap_words:] if overlap_words > 0 else []
                current_parts = [" ".join(last_words)] if last_words else []
                current_words = len(last_words)
        else:
            # если текущий буфер непуст, но добавление сегмента переполнит —
            # закрываем текущий chunk и не потребляем сегмент (оставим его на следующую итерацию)
            if current_parts:
                chunk_text = make_chunk_from_parts(current_parts)
                chunks.append(Chunk(f"{section_id}:{len(chunks)+1}", chunk_text, chunk_text, metadata))
                last_words = chunk_text.split()[-overlap_words:] if overlap_words > 0 else []
                current_parts = [" ".join(last_words)] if last_words else []
                current_words = len(last_words)
            else:
                # текущих частей нет, а сегмент сам слишком большой — вынести его один
                chunk_text = seg
                chunks.append(Chunk(f"{section_id}:{len(chunks)+1}", chunk_text, chunk_text, metadata))
                i += 1
                current_parts = []
                current_words = 0

    # после цикла — если что осталось в current_parts, оформить последний chunk
    if current_parts:
        chunk_text = make_chunk_from_parts(current_parts)
        if chunk_text.strip():
            chunks.append(Chunk(f"{section_id}:{len(chunks)+1}", chunk_text, chunk_text, metadata))

    return chunks