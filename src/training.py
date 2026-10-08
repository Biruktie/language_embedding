import random

from model import (
    forward,
    softmax,
    cross_entropy_loss,
    score_error,
    gradient_U,
    gradient_h,
    update_parameters
)


def mean_training_loss(E, U, training_pairs):
    total_loss = 0.0

    for center_id, target_id in training_pairs:
        _, scores = forward(E, U, center_id)

        loss = cross_entropy_loss(
            scores,
            target_id
        )

        total_loss += loss

    return total_loss / len(training_pairs)

def train(
    E,
    U,
    training_pairs,
    learning_rate,
    epochs
):
    loss_history = []

    initial_loss = mean_training_loss(
        E,
        U,
        training_pairs
    )

    print("Initial loss:", initial_loss)

    loss_history.append(initial_loss)

    for epoch in range(epochs):
        random.shuffle(training_pairs)

        for center_id, target_id in training_pairs:

            # Forward pass
            h, scores = forward(
                E,
                U,
                center_id
            )

            # Probabilities
            probabilities = softmax(scores)

            # Error
            errors = score_error(
                probabilities,
                target_id
            )

            # Gradients
            grad_U = gradient_U(
                h,
                errors
            )

            grad_h = gradient_h(
                U,
                errors
            )

            # Update
            update_parameters(
                E,
                U,
                center_id,
                grad_h,
                grad_U,
                learning_rate
            )

        epoch_loss = mean_training_loss(
            E,
            U,
            training_pairs
        )

        loss_history.append(epoch_loss)

        print(
            "Epoch",
            epoch + 1,
            "loss:",
            epoch_loss
        )

    return loss_history
