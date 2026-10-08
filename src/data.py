from pathlib import Path
import unicodedata

def separate_punctuation(line):
    punctuation = "።፥፤፦፧፨,!?;:()[]{}«»\"'"

    for mark in punctuation:
        line = line.replace(mark, f" {mark} ")

    return line

def preprocess_line(line):
    line = unicodedata.normalize("NFC", line)
    line = line.strip()

    if not line:
        return []

    cleaned_characters = []

    for character in line:
        if unicodedata.category(character) == "Co":
            continue

        cleaned_characters.append(character)

    line = "".join(cleaned_characters)

    line = line.replace("።.", "።")

    line = separate_punctuation(line)

    tokens = line.split()

    return tokens

def load_corpus(file_path):
    path = Path(file_path)

    text = path.read_text(encoding="utf-8")
    text = unicodedata.normalize("NFC", text)

    sentences = []

    for line in text.splitlines():
        tokens = preprocess_line(line)

        if tokens:
            sentences.append(tokens)

    return sentences

def is_punctuation(token):
    punctuation = {
        "።", "፥", "፤", "፦", "፧", "፨",
        ",", "!", "?", ";", ":", "(", ")",
        "[", "]", "{", "}", "«", "»", "\"", "'"
    }

    return token in punctuation

def build_vocabulary(sentences):
    word_to_id = {}
    id_to_word = []

    for sentence in sentences:
        for word in sentence:

            if is_punctuation(word):
                continue

            if word not in word_to_id:
                word_id = len(id_to_word)
                word_to_id[word] = word_id
                id_to_word.append(word)

    return word_to_id, id_to_word

def generate_training_pairs(sentences, word_to_id, window_size):
    pairs = []

    for sentence in sentences:
        for center_position in range(len(sentence)):
            center_word = sentence[center_position]

            if is_punctuation(center_word):
                continue

            center_id = word_to_id[center_word]

            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                context_position = center_position + offset

                if context_position < 0 or context_position >= len(sentence):
                    continue

                context_word = sentence[context_position]

                if is_punctuation(context_word):
                    continue

                context_id = word_to_id[context_word]

                pairs.append((center_id, context_id))

    return pairs

if __name__ == "__main__":
    sentences = load_corpus("data/corpus.txt")

    print("Number of sentences:", len(sentences))

    total_tokens = sum(len(sentence) for sentence in sentences)
    print("Total tokens:", total_tokens)

    all_words = []

    for sentence in sentences:
        for word in sentence:
            all_words.append(word)

    print("Unique tokens:", len(set(all_words)))

    private_use_characters = []

    for sentence in sentences:
        for word in sentence:
            for character in word:
                if unicodedata.category(character) == "Co":
                    private_use_characters.append(character)

    print("Private-use characters remaining:", len(private_use_characters))

    word_to_id, id_to_word = build_vocabulary(sentences)

    print("Vocabulary size:", len(word_to_id))

    window_size = 2

    training_pairs = generate_training_pairs(
        sentences,
        word_to_id,
        window_size
    )

    print("Window size:", window_size)
    print("Number of training pairs:", len(training_pairs))

    print("\nFirst 20 training pairs:")

    for center_id, context_id in training_pairs[:20]:
        print(
            id_to_word[center_id],
            "->",
            id_to_word[context_id]
        )