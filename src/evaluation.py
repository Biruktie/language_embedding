import math
import heapq
from collections import Counter

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

def nearest_neighbors(word, E, word_to_id, id_to_word, counts=None, top_k=3):
    if word not in word_to_id:
        raise KeyError(f"'{word}' is out of vocabulary")
    q = E[word_to_id[word]]
    sims = ((cosine_similarity(q, E[i]), i)
            for i in range(len(E)) if i != word_to_id[word])
    best = heapq.nlargest(top_k, sims)
    return [(id_to_word[i], round(s, 3), counts[id_to_word[i]] if counts else None)
            for s, i in best]

if __name__ == "__main__":

    freq = Counter(w for s in sentences for w in s if not is_punctuation(w))
    print("Hapax:", sum(1 for c in freq.values() if c == 1), "of", len(freq))
    print("Ending in ም:", sum(1 for w in freq if w.endswith("ም")), "types")
    for w in words_to_test: print(w, freq[w])
        # Test 1: cosine similarity
    # print("Cosine similarity tests:")

    # vector_a = [1.0, 0.0]
    # vector_b = [1.0, 0.0]

    # print(
    #     "Same vectors:",
    #     cosine_similarity(vector_a, vector_b)
    # )

    # vector_c = [1.0, 0.0]
    # vector_d = [0.0, 1.0]

    # print(
    #     "Orthogonal vectors:",
    #     cosine_similarity(vector_c, vector_d)
    # )

    # zero_vector = [0.0, 0.0]

    # print(
    #     "Zero vector:",
    #     cosine_similarity(vector_a, zero_vector)
    # )

    # # Test 2: unknown word
    # E = [
    #     [1.0, 0.0],
    #     [0.0, 1.0],
    #     [1.0, 1.0]
    # ]

    # word_to_id = {
    #     "word_a": 0,
    #     "word_b": 1,
    #     "word_c": 2
    # }

    # id_to_word = [
    #     "word_a",
    #     "word_b",
    #     "word_c"
    # ]

    # print("\nUnknown word test:")

    # known_result = nearest_neighbors(
    #     "word_a",
    #     E,
    #     word_to_id,
    #     id_to_word,
    #     top_k=2
    # )

    # unknown_result = nearest_neighbors(
    #     "not_in_vocabulary",
    #     E,
    #     word_to_id,
    #     id_to_word,
    #     top_k=2
    # )

    # print("Known word:", known_result)
    # print("Unknown word:", unknown_result)