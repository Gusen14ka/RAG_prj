from dataclasses import asdict
import json

from entities import Chunk

def save_chunks_json_as_dict(chunks: list[Chunk], path: str) -> None:
    chunks_by_id: dict[str, dict] = {}

    for c in chunks:
        cid = c.chunk_id
        if cid in chunks_by_id:
            raise ValueError(f"Duplicate chunk_id: {cid}")

        chunks_by_id[cid] = asdict(c)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks_by_id, f, ensure_ascii=False, indent=2)


def save_chunks_json_as_list(chunks: list[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            c_dict = asdict(c)
            f.write(json.dumps(c_dict, ensure_ascii=False) + "\n")