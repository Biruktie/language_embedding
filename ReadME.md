# Word Embeddings from Scratch

A small **Skip-Gram word embedding model implemented from scratch in Python** for the **Sebat Bet Gurage (Guragegna)** language.

No NumPy, PyTorch, TensorFlow, pretrained embeddings, or automatic differentiation are used.

## Features

- Unicode text preprocessing
- Vocabulary construction
- Center-context pair generation
- Skip-Gram forward pass
- Stable softmax and cross-entropy loss
- Manual backpropagation and gradient descent
- Cosine similarity and nearest-neighbor evaluation
- Unknown-word and zero-vector handling

## Project Structure

```text
language_embedding/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── main.py
├── data/
│   └── corpus.txt
└── README.md
```

- **`data.py`** — preprocessing, vocabulary, and training-pair generation
- **`model.py`** — Skip-Gram model and mathematical operations
- **`training.py`** — training loop and loss tracking
- **`evaluation.py`** — cosine similarity and nearest neighbors
- **`main.py`** — runs the complete experiment

## Dataset

The model is trained on **100 Sebat Bet Gurage sentences**.

After preprocessing:

- 1,879 total tokens
- 729 vocabulary words
- 4,674 training pairs

## Configuration

```text
Embedding dimension: 20
Window size:         2
Learning rate:       0.03
Epochs:              30
Random seed:         42
```

## Results

Training loss decreased from:

```text
Initial loss: 6.6607
Final loss:   2.6414
```

The model's learned embeddings are evaluated by finding the three nearest words using cosine similarity.

Because the corpus is intentionally small, the embeddings are primarily an educational demonstration rather than a production-quality language model.

## Run

From the project root:

```powershell
py .\src\main.py
```
