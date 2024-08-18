import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gensim
import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import SEQUENCE_LENGTH, EPOCHS, K_FOLD

word2vec_path = './data/GoogleNews-vectors-negative300.bin'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

csv = "/Users/atefeh/Desktop/fake news data/extracted/Final_version/bkup/data.csv"
data = pd.read_csv(csv)

def tokenize_and_vectorize(text, model, embedding_dim, max_length):
    tokens = word_tokenize(text)
    vectorized = [model[word] for word in tokens if word in model]
    if len(vectorized) > max_length:
        vectorized = vectorized[:max_length]
    else:
        vectorized.extend([np.zeros(embedding_dim)] * (max_length - len(vectorized)))
    return np.array(vectorized)

embedding_dim = 300  
max_length = SEQUENCE_LENGTH

embeddings = []
labels = []

for index, row in data.iterrows():
    text = row['Cleaned']
    if isinstance(text, str):
        embedding = tokenize_and_vectorize(text, word2vec, embedding_dim, max_length)
        embeddings.append(embedding)
        labels.append(row['label_fk'])

X = np.array(embeddings)
y = LabelEncoder().fit_transform(labels)

kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

fold = 1

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dropout(0.2, input_shape=(max_length, embedding_dim)))
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
