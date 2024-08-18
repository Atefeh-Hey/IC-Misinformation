import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import NE_EMB_SIZE_WITH_FEATURES, NE_LVL_FEATURES, NE_EMB_SIZE, DOC_EMB_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH_ALL, EPOCHS

import utils as utl

csv = "/Users/atefeh/Desktop/fake news data/extracted/Final_version/bkup/data.csv"
data = pd.read_csv(csv)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


def get_ne_embeddings_with_features(text, entities):
    """
    generating embeddings for the given text,
    identifying and extracting NE embeddings and discarding the rest,
    calculating NE level (word-level) features and concatenating them to the NE embs
    returning the final embeddings
    """

    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    outputs = bert_model(inputs)
    hidden_states = outputs.last_hidden_state

    entity_embeddings = []
    sentences = text.split('. ')  # Simple sentence splitting by period. Modify as needed.

    for entity in entities:
        # Find the start and end token indices for the entity in the text
        entity_tokens = tokenizer.tokenize(entity)
        entity_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

        start_idx = None
        for i in range(len(inputs['input_ids'][0]) - len(entity_ids) + 1):
            if inputs['input_ids'][0][i:i+len(entity_ids)].numpy().tolist() == entity_ids:
                start_idx = i
                break

        if start_idx is not None:
            end_idx = start_idx + len(entity_ids)
            entity_embedding = tf.reduce_mean(hidden_states[0][start_idx:end_idx], axis=0)

            # Find the sentence containing the NE to be used for feature extraction
            containing_sentence = next((sentence for sentence in sentences if entity in sentence), "")

            # Calculating word level features to the NE embeddings
            wrd_lvl_features = []

            sentiment_score = utl.calculate_sentiment_score(containing_sentence)    # 1 unit of feature score
            wrd_lvl_features.append(sentiment_score)

            emotion_scores = utl.calculate_emotion_scores(containing_sentence)    # 5 unit of feature score
            wrd_lvl_features.extend(emotion_scores)

            readability_scores = utl.calculate_readability_score(containing_sentence)    # 3 unit of feature score
            wrd_lvl_features.extend(readability_scores)

            # Concatenate word level features to the NE embeddings
            entity_embedding_with_features = tf.concat([
                entity_embedding, 
                tf.convert_to_tensor(wrd_lvl_features, dtype=tf.float32)
            ], axis=0)

            entity_embeddings.append(entity_embedding_with_features)

    return entity_embeddings


def get_ne_embeddings(text, entities):
    """
    for when we want no features to concat to the NE embeddings
    """
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    outputs = bert_model(inputs)
    hidden_states = outputs.last_hidden_state

    entity_embeddings = []

    for entity in entities:
        # Find the start and end token indices for the entity in the text
        entity_tokens = tokenizer.tokenize(entity)
        entity_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

        start_idx = None
        for i in range(len(inputs['input_ids'][0]) - len(entity_ids) + 1):
            if inputs['input_ids'][0][i:i+len(entity_ids)].numpy().tolist() == entity_ids:
                start_idx = i
                break

        if start_idx is not None:
            end_idx = start_idx + len(entity_ids)
            entity_embedding = tf.reduce_mean(hidden_states[0][start_idx:end_idx], axis=0)
            entity_embeddings.append(entity_embedding)

    return entity_embeddings


def process_segment(segment_name:str, article: dict):

    segment_text = article[segment_name]
    entities = utl.get_named_entities(segment_text)
    if NE_LVL_FEATURES:
        embeddings = get_ne_embeddings_with_features(segment_text, entities)
        ne_emb_size = NE_EMB_SIZE_WITH_FEATURES
    else:
        embeddings = get_ne_embeddings(segment_text, entities)
        ne_emb_size = NE_EMB_SIZE
    
    # Average wrd-lvl embeddings to form seg-level embs
    if len(embeddings) > 0:
        seg_embedding = np.mean([e.numpy() for e in embeddings], axis=0)
    else:
        # For when there is no NE embeddings
        seg_embedding = np.zeros((ne_emb_size,))  # number of NE embs after enriching with wrd-lvl features

    # Calculate segment levl features and concatenate to seg embeddings
    seg_lvl_scores = []

    ne_semantic_similarity_score = utl.calculate_semantic_similarity(embeddings)                                      #1 unit to be added to the segment embs
    seg_lvl_scores.append(ne_semantic_similarity_score)

    topics_similarity_score = utl.get_segment_topics_similarity_score(segment_name, article)                          #1 unit to be added to the segment embs
    seg_lvl_scores.append(topics_similarity_score)
    readability_scores = utl.get_segment_readability_scores(segment_name, article)                                    #3 unit to be added to the segment embs
    seg_lvl_scores.extend(readability_scores)

    seg_embedding_with_features = tf.concat([
        tf.convert_to_tensor(seg_embedding, dtype=tf.float32),
        tf.convert_to_tensor(seg_lvl_scores, dtype=tf.float32)
    ], axis=0)

    return seg_embedding_with_features


def process_article(article):
    title_emb = process_segment(segment_name='Title', article=article)
    fp_emb = process_segment(segment_name='FP', article=article)
    body_emb = process_segment(segment_name='body', article=article)

    artilce_txt = article['Cleaned']

    doc_lvl_scores = []
    doc_topic_similarity_score = utl.get_doc_topics_similarity_score(article)                           #1 unit to be added to the article embs
    doc_lvl_scores.append(doc_topic_similarity_score)
    doc_ne_embeddings_similarity_score = utl.calculate_doc_ne_semantic_similarity(artilce_txt)          #1 unit to be added to the article embs
    doc_lvl_scores.append(doc_ne_embeddings_similarity_score)

    doc_lvl_scores = tf.convert_to_tensor(doc_lvl_scores, dtype=tf.float32)
    # Concatenate segment embeddings + doc lvl features
    document_embedding = np.concatenate([title_emb, fp_emb, body_emb, doc_lvl_scores])

    return document_embedding


# #### Training data ####

embeddings = []
labels = []

for index, row in data.iterrows():
    embedding = process_article(row)
    embedding_np = embedding.numpy()[0]  # Remove the batch dimension
    if embedding_np.shape[0] < DOC_EMB_SIZE:    #this is unneccessary as the length of embeddings is fixed, but not bad for ensuring the length
        # Pad to max length
        padding = np.zeros((DOC_EMB_SIZE - embedding_np.shape[0], embedding_np.shape[1]))
        embedding_np = np.vstack([embedding_np, padding])
    elif embedding_np.shape[0] > DOC_EMB_SIZE:
        # Truncate to max length
        embedding_np = embedding_np[:DOC_EMB_SIZE, :]

    embeddings.append(embedding_np)
    labels.append(row['label_fk'])

X = np.array(embeddings)
y = LabelEncoder().fit_transform(labels)

# Initialize KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

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

    embedding_dim = DOC_EMB_SIZE
    input_length = SEQUENCE_LENGTH_ALL   

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dropout(0.2, input_shape=(input_length, embedding_dim)))
    model.add(tf.keras.layers.Conv1D(128, 4, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calculate metrics for this fold
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Store metrics
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    fold += 1

# Calculate average metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print("\n10-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")