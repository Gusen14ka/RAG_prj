import json

from entities import (
    Chunk,
    ChunkMetaData,
)

"""
Утилита для считывания из .jsonl - каждая строка один словарь
"""
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

"""
Утилита для считывания из .json - свободный формат в виде json
"""
def load_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



def _chunk_from_dict(data: dict) -> Chunk:
    metadata = data.get("metadata", {})
    return Chunk(
        chunk_id=data["chunk_id"],
        clean_text=data["clean_text"],
        raw_text=data["raw_text"],
        metadata=ChunkMetaData(
            section=metadata.get("section", ""),
            subsection=metadata.get("subsection", ""),
        ),
    )

"""
Утилита для считания чанков из .json - общий словарь с ключами chunk_id
"""
def load_chunks_json_as_dict(path: str) -> dict[str, Chunk]:
    raw_chunks = load_json(path)
    return {
        chunk_id: _chunk_from_dict(chunk_data)
        for chunk_id, chunk_data in raw_chunks.items()
    }

"""
Утилита для считания чанков из .jsonl - каждая строка - словарь-чанк
"""
def load_chunks_jsonl_as_list(path: str) -> list[Chunk]:
    raw_chunks = load_jsonl(path)
    return [_chunk_from_dict(chunk_data) for chunk_data in raw_chunks]

