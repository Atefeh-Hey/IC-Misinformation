"""
This is the baseline CNN model using the GloVe encoding woth 100 dimensions as used reported achieving high accuracy in state of the art approaches.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import SEQUENCE_LENGTH, EPOCHS, K_FOLD


def custom_pad_sequences(sequences, maxlen):
    """
    as incompatibility of the tensorflow and keras for making use of keras padsequence()
    """
    return np.array([
        np.pad(seq, (maxlen - len(seq), 0), mode='constant', constant_values=0)
        if len(seq) < maxlen else np.array(seq[:maxlen])
        for seq in sequences
    ])


csv = "/Users/atefeh/Desktop/fake news data/extracted/Final_version/bkup/data.csv"
data = pd.read_csv(csv)

glove_path = './data/glove.6B.100d.txt'
embedding_dim = 100

embedding_index = {}
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

texts = data['Cleaned'].astype(str).tolist()
labels = data['label_fk'].tolist()

max_length = SEQUENCE_LENGTH
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index



X = custom_pad_sequences(sequences, max_length)
y = LabelEncoder().fit_transform(labels)

vocab_size = len(word_index) + 1  
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

fold = 1

for train_index, val_index in kf.split(X):
    print(f"Training fold {fold}...")
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size,
                                        output_dim=embedding_dim,
                                        weights=[embedding_matrix],
                                        # input_length=max_length,
                                        trainable=False))

    model.add(tf.keras.layers.Conv1D(128, 4, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    fold += 1

mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print("\n10-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")
