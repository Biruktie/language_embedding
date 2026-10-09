import random

from data import build_vocabulary, generate_training_pairs, load_corpus
from model import initialize_parameters
from training import train
from evaluation import nearest_neighbors


def run_experiment(sentences, word_to_id, id_to_word, window_size):
    # Reset the seed so both experiments start with the same weights.
    random.seed(42)

    embedding_dim = 20
    learning_rate = 0.03
    epochs = 30

    pairs = generate_training_pairs(
        sentences,
        word_to_id,
        window_size
    )

    E, U = initialize_parameters(
        len(word_to_id),
        embedding_dim
    )

    print(f"\n===== WINDOW SIZE {window_size} =====")
    print("Training pairs:", len(pairs))

    losses = train(
        E,
        U,
        pairs,
        learning_rate,
        epochs
    )

    print("Initial loss:", losses[0])
    print("Final loss:", losses[-1])

    return E


def main():
    sentences = load_corpus("data/corpus.txt")
    word_to_id, id_to_word = build_vocabulary(sentences)

    words_to_compare = [
        "ትንም",
        "ይፍቴ",
        "እግዘር",
        "አፈርም",
        "ሰሜም"
    ]

    E_window_2 = run_experiment(
        sentences, word_to_id, id_to_word, window_size=2
    )

    E_window_3 = run_experiment(
        sentences, word_to_id, id_to_word, window_size=3
    )

    print("\n===== NEAREST-NEIGHBOR COMPARISON =====")

    for word in words_to_compare:
        print(f"\nQuery word: {word}")

        neighbors_2 = nearest_neighbors(
            word, E_window_2, word_to_id, id_to_word, top_k=3
        )

        neighbors_3 = nearest_neighbors(
            word, E_window_3, word_to_id, id_to_word, top_k=3
        )

        print("Window 2:", neighbors_2)
        print("Window 3:", neighbors_3)


if __name__ == "__main__":
    main()