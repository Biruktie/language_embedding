import json
from pathlib import Path


def save_model(path, E, U, word_to_id, id_to_word):
    """Save embeddings, output weights, and vocabulary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "E": E,
        "U": U,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(model_data, file, ensure_ascii=False)


def load_model(path):
    """Load the saved model and vocabulary from JSON."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        model_data = json.load(file)

    E = model_data["E"]
    U = model_data["U"]
    word_to_id = model_data["word_to_id"]
    id_to_word = model_data["id_to_word"]

    return E, U, word_to_id, id_to_word