import random

from data import build_vocabulary, generate_training_pairs, load_corpus
from model import (
    initialize_parameters,
    forward,
    softmax,
    cross_entropy_loss,
    score_error,
    gradient_U,
    gradient_h
)


EPSILON = 0.00001


def loss_for_pair(E, U, center_id, target_id):
    _, scores = forward(E, U, center_id)
    return cross_entropy_loss(scores, target_id)


def numerical_gradient(get_value, set_value, loss_function):
    original_value = get_value()

    set_value(original_value + EPSILON)
    loss_plus = loss_function()

    set_value(original_value - EPSILON)
    loss_minus = loss_function()

    set_value(original_value)

    return (loss_plus - loss_minus) / (2 * EPSILON)


def main():
    random.seed(42)

    sentences = load_corpus("data/corpus.txt")
    word_to_id, id_to_word = build_vocabulary(sentences)

    pairs = generate_training_pairs(
        sentences,
        word_to_id,
        window_size=2
    )

    E, U = initialize_parameters(
        len(word_to_id),
        embedding_dim=20
    )

    center_id, target_id = pairs[0]

    # Calculate analytic gradients using the original parameters.
    h, scores = forward(E, U, center_id)
    probabilities = softmax(scores)
    errors = score_error(probabilities, target_id)

    analytic_grad_U = gradient_U(h, errors)
    analytic_grad_h = gradient_h(U, errors)

    # Check one coordinate of the center word's embedding.
    embedding_coordinate = 0

    numerical_E = numerical_gradient(
        lambda: E[center_id][embedding_coordinate],
        lambda value: E[center_id].__setitem__(embedding_coordinate, value),
        lambda: loss_for_pair(E, U, center_id, target_id)
    )

    analytic_E = analytic_grad_h[embedding_coordinate]

    # Check one coordinate of the output-weight matrix.
    output_row = 0
    output_column = target_id

    numerical_U = numerical_gradient(
        lambda: U[output_row][output_column],
        lambda value: U[output_row].__setitem__(output_column, value),
        lambda: loss_for_pair(E, U, center_id, target_id)
    )

    analytic_U = analytic_grad_U[output_row][output_column]

    print("Center word:", id_to_word[center_id])
    print("Target word:", id_to_word[target_id])

    print("\nEmbedding gradient check:")
    print("Analytic:", analytic_E)
    print("Numerical:", numerical_E)
    print("Absolute difference:", abs(analytic_E - numerical_E))

    print("\nOutput-weight gradient check:")
    print("Analytic:", analytic_U)
    print("Numerical:", numerical_U)
    print("Absolute difference:", abs(analytic_U - numerical_U))


if __name__ == "__main__":
    main()