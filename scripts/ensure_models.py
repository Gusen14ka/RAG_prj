import os
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

"""
Базовый скрипт для загрузки моделей по config файлу
"""

def ensure_model(name, path):
    if os.path.exists(path) and os.listdir(path):
        print(f"[OK] {path} already exists")
        return

    print(f"[DOWNLOAD] {name}")
    if name == "DiTy/cross-encoder-russian-msmarco" or name == "BAAI/bge-reranker-v2-m3":
        model = CrossEncoder(name)
    else:
        model = SentenceTransformer(name)
    model.save(path)
    print(f"[SAVED] {path}")


def main():
    with open("config/models_config.json", "r") as f:
        config = json.load(f)

    for m in config["models"]:
        ensure_model(m["name"], m["path"])


if __name__ == "__main__":
    main()