import random

from data import (
    build_vocabulary,
    generate_training_pairs,
    load_corpus
)

from model import initialize_parameters

from training import train

from evaluation import nearest_neighbors


def main():
    random.seed(42)

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    embedding_dim = 20
    window_size = 2
    learning_rate = 0.03
    epochs = 30

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    sentences = load_corpus("data/corpus.txt")

    print("===== CORPUS =====")
    print("Number of sentences:", len(sentences))

    total_tokens = 0

    for sentence in sentences:
        total_tokens += len(sentence)

    print("Total tokens:", total_tokens)

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    word_to_id, id_to_word = build_vocabulary(sentences)

    print("Vocabulary size:", len(word_to_id))

    # --------------------------------------------------------
    # Training pairs
    # --------------------------------------------------------

    training_pairs = generate_training_pairs(
        sentences,
        word_to_id,
        window_size
    )

    print("Window size:", window_size)
    print("Training pairs:", len(training_pairs))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    vocab_size = len(word_to_id)

    E, U = initialize_parameters(
        vocab_size,
        embedding_dim
    )

    print("\n===== MODEL =====")
    print("Vocabulary size:", vocab_size)
    print("Embedding dimension:", embedding_dim)
    print("E shape:", len(E), "x", len(E[0]))
    print("U shape:", len(U), "x", len(U[0]))

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n===== TRAINING =====")

    loss_history = train(
        E,
        U,
        training_pairs,
        learning_rate,
        epochs
    )

    print("\nTraining complete.")
    print("Initial loss:", loss_history[0])
    print("Final loss:", loss_history[-1])

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n===== EVALUATION =====")

    words_to_test = [
        "ትንም",
        "ይፍቴ",
        "እግዘር",
        "አፈርም",
        "ሰሜም"
    ]

    print("\nNearest neighbors:")

    for word in words_to_test:
        neighbors = nearest_neighbors(
            word,
            E,
            word_to_id,
            id_to_word,
            top_k=3
        )

        print(word, "->", neighbors)

    # --------------------------------------------------------
    # Unknown word test
    # --------------------------------------------------------

    unknown_word = "ይህ_በቃ_አይገኝም"

    unknown_neighbors = nearest_neighbors(
        unknown_word,
        E,
        word_to_id,
        id_to_word,
        top_k=3
    )

    print("\nUnknown word test:")
    print(unknown_word, "->", unknown_neighbors)


if __name__ == "__main__":
    main()