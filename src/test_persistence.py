import random

from data import build_vocabulary, load_corpus
from model import initialize_parameters
from persistence import save_model, load_model


def main():
    random.seed(42)

    sentences = load_corpus("data/corpus.txt")
    word_to_id, id_to_word = build_vocabulary(sentences)

    E, U = initialize_parameters(
        len(word_to_id),
        20
    )

    save_model(
        "models/guragegna_model.json",
        E,
        U,
        word_to_id,
        id_to_word
    )

    loaded_E, loaded_U, loaded_word_to_id, loaded_id_to_word = (
        load_model("models/guragegna_model.json")
    )

    assert loaded_E == E
    assert loaded_U == U
    assert loaded_word_to_id == word_to_id
    assert loaded_id_to_word == id_to_word

    print("Save and reload: passed")
    print("Embedding matrix:", len(loaded_E), "x", len(loaded_E[0]))
    print("Output matrix:", len(loaded_U), "x", len(loaded_U[0]))
    print("Vocabulary size:", len(loaded_word_to_id))


if __name__ == "__main__":
    main()