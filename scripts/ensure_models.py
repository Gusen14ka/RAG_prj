import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Скрипт для загрузки моделей по config файлу.
Запускать из корня проекта: python scripts/ensure_models.py
"""

# Корень проекта — папка на уровень выше этого скрипта
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "models_config.json"


def ensure_model(type: str, name: str, path: str) -> None:
    model_path = PROJECT_ROOT / path
    if model_path.exists() and any(model_path.iterdir()):
        print(f"[OK] {model_path} already exists")
        return

    model_path.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {name}")
    if type == "crossencoder":
        model = CrossEncoder(name)
        model.save(str(model_path))

    elif type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))

    elif type == "biencoder":
        model = SentenceTransformer(name)
        model.save(str(model_path))

    else:
        raise ValueError(f"Unknown model type: {type}")
    
    print(f"[SAVED] {model_path}")



def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    for m in config["models"]:
        ensure_model(m["type"], m["name"], m["path"])


if __name__ == "__main__":
    main()