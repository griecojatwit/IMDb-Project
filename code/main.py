import os
import numpy as np
import pandas as pd
import re
import random
from collections import Counter

def clean_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

def vectorize(text, word_index):
    x = np.zeros(len(word_index))
    for w in text.split():
        if w in word_index:
            x[word_index[w]] += 1
    return x

def vectorize_binary(text, word_index):
    x = np.zeros(len(word_index))
    for w in set(text.split()):
        if w in word_index:
            x[word_index[w]] = 1
    return x

def load_and_preprocess_data(csv_path, vocab_size=5000):
    df = pd.read_csv(csv_path)
    df['review'] = df['review'].astype(str).apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    train_df, test_df = df.iloc[:25000], df.iloc[25000:50000]

    word_counts = Counter(" ".join(train_df['review']).split())
    vocab = [w for w, _ in word_counts.most_common(vocab_size)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    X_train = np.stack([vectorize(r, word2idx) for r in train_df['review']])
    y_train = train_df['sentiment'].values.reshape(-1, 1)
    X_test = np.stack([vectorize(r, word2idx) for r in test_df['review']])
    y_test = test_df['sentiment'].values.reshape(-1, 1)

    return X_train, y_train, X_test, y_test, len(vocab), word2idx, train_df, test_df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_simple_nn(X_train, y_train, input_dim, hidden_dim=32, epochs=10, lr=0.01, batch_size=128):
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * 0.01
    b2 = np.zeros((1, 1))
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            Xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]

            # Forward pass
            Z1 = Xb @ W1 + b1
            A1 = np.tanh(Z1)
            Z2 = A1 @ W2 + b2
            A2 = sigmoid(Z2)

            # Compute loss (for first batch only, for monitoring)
            if start == 0:
                loss = -np.mean(yb * np.log(A2 + 1e-8) + (1 - yb) * np.log(1 - A2 + 1e-8))

            # Backward pass
            dZ2 = A2 - yb
            dW2 = A1.T @ dZ2 / len(Xb)
            db2 = np.mean(dZ2, axis=0, keepdims=True)
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * (1 - A1 ** 2)
            dW1 = Xb.T @ dZ1 / len(Xb)
            db1 = np.mean(dZ1, axis=0, keepdims=True)

            # Update weights
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

def evaluate_simple_nn(X, y, W1, b1, W2, b2):
    A1 = np.tanh(X @ W1 + b1)
    A2 = sigmoid(A1 @ W2 + b2)
    acc = np.mean((A2 > 0.5) == y)
    print("Test Accuracy:", acc)
    return acc

def add_typo(word):
    if len(word) < 3:
        return word
    i = random.randint(1, len(word)-2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]

def noisy_review(review, typo_prob=0.1):
    words = review.split()
    noisy = []
    for w in words:
        if random.random() < typo_prob:
            noisy.append(add_typo(w))
        else:
            noisy.append(w)
    return " ".join(noisy)

if __name__ == "__main__":
    csv_path = "data/IMDB Dataset.csv"

    X_train, y_train, X_test, y_test, input_dim, word_index, train_df, test_df = load_and_preprocess_data(csv_path, vocab_size=5000)

    W1, b1, W2, b2 = train_simple_nn(X_train, y_train, input_dim, hidden_dim=32, epochs=25, lr=0.01, batch_size=64)
    print("\nClean test set evaluation:")
    evaluate_simple_nn(X_test, y_test, W1, b1, W2, b2)

    influence = np.dot(W1, W2).flatten()
    index_word = {v: k for k, v in word_index.items()}
    sorted_indices = np.argsort(influence)
    top_pos = [(index_word.get(i, ''), float(influence[i])) for i in sorted_indices[-10:]]
    top_neg = [(index_word.get(i, ''), float(influence[i])) for i in sorted_indices[:10]]

    print("\nTop positive words:")
    for rank, (word, score) in enumerate(reversed(top_pos), 1):
        print(f"{rank:2d}. {word:12s} {score: .2f}")

    print("\nTop negative words:")
    for rank, (word, score) in enumerate(top_neg, 1):
        print(f"{rank:2d}. {word:12s} {score: .2f}")

    test_df['noisy_review'] = test_df['review'].apply(lambda x: noisy_review(x, typo_prob=0.15))
    X_test_noisy = np.stack([vectorize(r, word_index) for r in test_df['noisy_review']])
    evaluate_simple_nn(X_test_noisy, y_test, W1, b1, W2, b2)

    X_train_bin = np.stack([vectorize_binary(r, word_index) for r in train_df['review']])
    X_test_bin = np.stack([vectorize_binary(r, word_index) for r in test_df['review']])
    W1_bin, b1_bin, W2_bin, b2_bin = train_simple_nn(X_train_bin, y_train, input_dim, hidden_dim=32, epochs=25, lr=0.01, batch_size=64)
    evaluate_simple_nn(X_test_bin, y_test, W1_bin, b1_bin, W2_bin, b2_bin)
