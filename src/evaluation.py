import math
import heapq


def cosine_similarity(vector_a, vector_b):
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(vector_a)):
        dot_product += vector_a[i] * vector_b[i]
        norm_a += vector_a[i] * vector_a[i]
        norm_b += vector_b[i] * vector_b[i]

    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def nearest_neighbors(
    word,
    E,
    word_to_id,
    id_to_word,
    counts=None,
    top_k=3
):
    if word not in word_to_id:
        return []

    word_id = word_to_id[word]
    query_vector = E[word_id]

    similarities = (
        (cosine_similarity(query_vector, E[i]), i)
        for i in range(len(E))
        if i != word_id
    )

    best = heapq.nlargest(top_k, similarities)

    return [
        (
            id_to_word[i],
            round(similarity, 3),
            counts[id_to_word[i]] if counts else None
        )
        for similarity, i in best
    ]