import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

class DummyFile:
    def write(self, x): pass
    def flush(self): pass
    def close(self): pass

sys.stdout = DummyFile()
sys.stderr = DummyFile()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, InputLayer  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from sklearn.model_selection import train_test_split
import random
import re

def clean_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

def add_typo(word):
    if len(word) < 3:
        return word
    i = random.randint(1, len(word)-2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]

def noisy_text(text, typo_prob=0.15):
    words = text.split()
    noisy = []
    for w in words:
        if random.random() < typo_prob:
            noisy.append(add_typo(w))
        else:
            noisy.append(w)
    return " ".join(noisy)

def prepare_data(csv_path, num_words=10000, maxlen=200, test_size=0.5):
    df = pd.read_csv(csv_path)
    df['review'] = df['review'].astype(str).apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    texts = df['review'].tolist()
    labels = df['sentiment'].values

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=maxlen)
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return df, tokenizer, x_train, x_test, y_train, y_test, texts

def train_embedding_model(x_train, y_train, num_words=10000, maxlen=200):
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=16, input_length=maxlen),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train, epochs=10, batch_size=512, validation_split=0.5, verbose=2
    )
    return model

def evaluate_model(model, x_test, y_test, label="clean"):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy ({label}): {test_accuracy:.4f}")

def get_noisy_sequences(texts, tokenizer, maxlen=200, typo_prob=0.15):
    noisy_texts = [noisy_text(text, typo_prob=typo_prob) for text in texts]
    noisy_sequences = tokenizer.texts_to_sequences(noisy_texts)
    x_noisy = pad_sequences(noisy_sequences, maxlen=maxlen)
    return x_noisy

def print_top_words(model, tokenizer, n=10):
    embedding_matrix = model.layers[0].get_weights()[0]
    dense_weights = model.layers[-1].get_weights()[0].flatten()
    word_influence = np.dot(embedding_matrix, dense_weights)
    index_word = {v: k for k, v in tokenizer.word_index.items() if v < len(word_influence)}
    sorted_indices = np.argsort(word_influence)
    top_pos = [(index_word.get(i, ''), float(word_influence[i])) for i in sorted_indices[-n:]]
    top_neg = [(index_word.get(i, ''), float(word_influence[i])) for i in sorted_indices[:n]]

    print("\nTop positive words:")
    for rank, (word, score) in enumerate(reversed(top_pos), 1):
        print(f"{rank:2d}. {word:12s} {score: .2f}")

    print("\nTop negative words:")
    for rank, (word, score) in enumerate(top_neg, 1):
        print(f"{rank:2d}. {word:12s} {score: .2f}")

def vectorize_binary(text, word_index, vocab_size):
    vec = np.zeros(vocab_size)
    for w in set(text.split()):
        idx = word_index.get(w)
        if idx is not None and idx < vocab_size:
            vec[idx] = 1
    return vec

def train_binary_bow(df, tokenizer, x_train_len, x_test_len, y_train, y_test, vocab_size=10000):
    word_index = tokenizer.word_index
    X_train_bin = np.stack([
        vectorize_binary(text, word_index, vocab_size)
        for text in df['review'].iloc[:x_train_len]
    ])
    X_test_bin = np.stack([
        vectorize_binary(text, word_index, vocab_size)
        for text in df['review'].iloc[x_train_len:x_train_len + x_test_len]
    ])

    bin_model = Sequential([
        InputLayer(input_shape=(vocab_size,)), Dense(16, activation='relu'), Dense(1, activation='sigmoid')
    ])
    bin_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    bin_model.fit(X_train_bin, y_train, epochs=10, batch_size=512, validation_split=0.5, verbose=2)
    bin_loss, bin_acc = bin_model.evaluate(X_test_bin, y_test, verbose=2)
    print(f"Binary BOW test accuracy: {bin_acc:.4f}")

if __name__ == "__main__":
    csv_path = "data/IMDB Dataset.csv"

    df, tokenizer, x_train, x_test, y_train, y_test, texts = prepare_data(csv_path)
    model = train_embedding_model(x_train, y_train)
    evaluate_model(model, x_test, y_test, label="clean")

    test_texts = df['review'].iloc[len(x_train):len(x_train)+len(x_test)].tolist()
    x_test_noisy = get_noisy_sequences(test_texts, tokenizer)
    evaluate_model(model, x_test_noisy, y_test, label="noisy, typo_prob=0.15")

    print_top_words(model, tokenizer, n=10)
    train_binary_bow(df, tokenizer, len(x_train), len(x_test), y_train, y_test)
