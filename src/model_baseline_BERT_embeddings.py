import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import NE_EMB_SIZE, SEQUENCE_LENGTH, EPOCHS, K_FOLD

csv = "/Users/atefeh/Desktop/fake news data/extracted/Final_version/bkup/data.csv"
data = pd.read_csv(csv)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=SEQUENCE_LENGTH)
    outputs = bert_model(inputs)
    return outputs.last_hidden_state

embeddings = []
labels = []

for index, row in data.iterrows():
    text = row['Cleaned']
    label = row['label_fk']

    if not isinstance(text, str):
        print(f"Problematic row index: {index}")
        print(f"ID: {row['id']}")
        print(f"Type of 'Cleaned': {type(text)}")
        print(f"Value of 'Cleaned': {text}")
        continue
     
    embedding = get_bert_embeddings([text])
    # Convert to numpy array and ensure consistent shape
    embedding_np = embedding.numpy()[0]  # Remove the batch dimension
    if embedding_np.shape[0] < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - embedding_np.shape[0], embedding_np.shape[1]))
        embedding_np = np.vstack([embedding_np, padding])
    elif embedding_np.shape[0] > SEQUENCE_LENGTH:
        # Truncate to max length
        embedding_np = embedding_np[:SEQUENCE_LENGTH, :]
    embeddings.append(embedding_np)
    labels.append(label)

X = np.array(embeddings)
y = LabelEncoder().fit_transform(labels)

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
    
    # Convert to tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    embedding_dim = NE_EMB_SIZE
    input_length = SEQUENCE_LENGTH   

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dropout(0.2, input_shape=(input_length, embedding_dim)))
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
