import json
"""
Утилита для считания чанков из .jsonl - каждая строка один чанк-словарь
"""
def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj)
    return chunks

"""
Утилита для считания чанков из .json - общий словарь с ключами chunk_id
"""
def load_chunks_with_key(path):
    """Load chunks saved with `save_chunks_as_json`.

    The file is expected to contain a single JSON object where keys are `chunk_id`.
    Returns a list of chunk dicts (values of that object).
    """

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError("JSON не содержит словарь (dict)")
    
    return data

