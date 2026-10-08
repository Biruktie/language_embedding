import random
import math


def initialize_parameters(vocab_size, embedding_dim):
    E = []
    U = []

    for _ in range(vocab_size):
        row = []

        for _ in range(embedding_dim):
            row.append(random.uniform(-0.5, 0.5))

        E.append(row)

    for _ in range(embedding_dim):
        row = []

        for _ in range(vocab_size):
            row.append(random.uniform(-0.5, 0.5))

        U.append(row)

    return E, U

def forward(E, U, center_id):
    h = E[center_id]

    scores = []

    for j in range(len(U[0])):
        score = 0.0

        for k in range(len(h)):
            score += h[k] * U[k][j]

        scores.append(score)

    return h, scores

def softmax(scores):
    max_score = max(scores)

    exp_scores = []

    for score in scores:
        exp_scores.append(math.exp(score - max_score))

    total = sum(exp_scores)

    probabilities = []

    for value in exp_scores:
        probabilities.append(value / total)

    return probabilities

def cross_entropy_loss(scores, target_id):
    max_score = max(scores)

    exp_scores = []

    for score in scores:
        exp_scores.append(math.exp(score - max_score))

    total = sum(exp_scores)

    loss = (max_score - scores[target_id]) + math.log(total)

    return loss

def score_error(probabilities, target_id):
    errors = []

    for j in range(len(probabilities)):
        error = probabilities[j]

        if j == target_id:
            error -= 1.0

        errors.append(error)

    return errors

def gradient_U(h, errors):
    grad = []

    for k in range(len(h)):
        row = []

        for j in range(len(errors)):
            row.append(h[k] * errors[j])

        grad.append(row)

    return grad

def gradient_h(U, errors):
    grad = []

    for k in range(len(U)):
        value = 0.0

        for j in range(len(errors)):
            value += U[k][j] * errors[j]

        grad.append(value)

    return grad

def update_parameters(E, U, center_id, grad_h, grad_U, learning_rate):
    # Update the selected embedding row
    for k in range(len(E[center_id])):
        E[center_id][k] -= learning_rate * grad_h[k]

    # Update every element of U
    for k in range(len(U)):
        for j in range(len(U[k])):
            U[k][j] -= learning_rate * grad_U[k][j]

