import random

from data import (
    build_vocabulary,
    generate_training_pairs,
    load_corpus
)

from model import initialize_parameters

from training import train

from evaluation import cosine_similarity, nearest_neighbors


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
    # Evaluation tests
    # --------------------------------------------------------

    print("\n===== EVALUATION TESTS =====")

    # Test 1: Identical vectors should have cosine similarity 1.
    same_similarity = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0]
    )

    print("Identical vectors:", same_similarity)
    assert abs(same_similarity - 1.0) < 1e-5

    # Test 2: Orthogonal vectors should have cosine similarity 0.
    orthogonal_similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0]
    )

    print("Orthogonal vectors:", orthogonal_similarity)
    assert abs(orthogonal_similarity) < 1e-5

    # Test 3: A zero-norm vector must not cause a crash.
    zero_similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 0.0]
    )

    print("Zero-norm vector:", zero_similarity)
    assert zero_similarity == 0.0

    print("Cosine similarity tests: passed")

    # Test 4: Query five words from the trained vocabulary.
    words_to_test = [
        "ትንም",
        "ይፍቴ",
        "እግዘር",
        "አፈርም",
        "ሰሜም"
    ]

    print("\n===== NEAREST NEIGHBORS =====")

    for word in words_to_test:
        assert word in word_to_id, f"Test word missing: {word}"

        neighbors = nearest_neighbors(
            word,
            E,
            word_to_id,
            id_to_word,
            top_k=3
        )

        print(f"{word} -> {neighbors}")

        # Each query should have three neighbors in this vocabulary.
        assert len(neighbors) == 3

        # The query word must not appear among its own neighbors.
        neighbor_words = [item[0] for item in neighbors]
        assert word not in neighbor_words

    print("Nearest-neighbor tests: passed")

    # Test 5: Unknown words must be handled safely.
    unknown_word = "not_in_vocabulary"

    unknown_result = nearest_neighbors(
        unknown_word,
        E,
        word_to_id,
        id_to_word,
        top_k=3
    )

    print("\nUnknown word result:", unknown_result)

    assert unknown_result == []
    print("Unknown-word handling: passed")


if __name__ == "__main__":
    main()