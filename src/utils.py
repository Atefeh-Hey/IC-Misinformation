from afinn import Afinn
import numpy as np
from textstat import textstat
from collections import defaultdict
from nrclex import NRCLex
import spacy
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from scipy.spatial.distance import cosine


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


nlp = spacy.load("en_core_web_sm")      #MUST install the model for the first time to be able to load it. USE THIS: python -m spacy download en_core_web_sm

# ### for emotion NRC lexicon ###

# Load the NRC Emotion Lexicon
lexicon_path = './data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt' 
lexicon = pd.read_csv(lexicon_path, sep='\t', header=None, names=['word', 'emotion', 'value'])

# Define emotions of interest
emotions_of_interest = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Filter lexicon for emotions of interest
filtered_lexicon = lexicon[lexicon['emotion'].isin(emotions_of_interest)]

# Create a dictionary to map words to emotion scores
emotion_dict = defaultdict(lambda: defaultdict(int))
for _, row in filtered_lexicon.iterrows():
    emotion_dict[row['word']][row['emotion']] = row['value']
# ###


def calculate_sentiment_score(text: str) -> float:
    afinn = Afinn()
    score = afinn.score(text)

    score = max(-5, min(score, 5))      #to ensure that the score is betwen -5 to 5

    # normalise to 0 to1 range
    normalized_score = (score + 5) / 10 
    return np.clip(normalized_score, 0, 1)


def normalise_readability(score: float, fres:bool = False) -> float:
    if fres:
        min_value = 0
        max_value = 100
        score = 100 - score        # As FRES shows higher score for easier text to comprehend
    else:
        min_value = 5
        max_value = 18

    if score < min_value:
        score = min_value
    elif score > max_value:
        score = max_value

    # Normalize the value
    normalized_value = (score - min_value) / (max_value - min_value)
    normalized_value = round(normalized_value, 2)
    # Return the normalized value
    return normalized_value


def calculate_readability_score(text):

    fres_score = textstat.flesch_reading_ease(text)
    fres_score_normalised = normalise_readability(score=fres_score, fres=True)

    fkg_score = textstat.flesch_kincaid_grade(text)
    fkg_score_normalised = normalise_readability(score=fkg_score)

    fog_score = textstat.gunning_fog(text)
    fog_score_normalised = normalise_readability(score=fog_score)

    readability_scores = [fres_score_normalised, fkg_score_normalised, fog_score_normalised]

    return readability_scores


def calculate_emotion_scores_dict(text: str) -> dict:
    """
    to test how emotion scoring works, it outputs the scores for each emotion as part of a dict
    """

    tokens = word_tokenize(text.lower())
    emotion_scores = {emotion: 0 for emotion in emotions_of_interest}
    total_words = len(tokens)
    
    if total_words == 0:
        return {emotion: 0 for emotion in emotions_of_interest}
    
    # Calculate scores
    for token in tokens:
        if token in emotion_dict:
            for emotion in emotions_of_interest:
                emotion_scores[emotion] += emotion_dict[token][emotion]
    
    # normalise scores
    max_scores = {emotion: max(emotion_scores.values()) for emotion in emotions_of_interest}
    normalized_scores = {emotion: emotion_scores[emotion] / (max_scores[emotion] if max_scores[emotion] > 0 else 1)
                         for emotion in emotions_of_interest}
    
    return normalized_scores


def calculate_emotion_scores(text: str) -> list:
    """
    this is the main method used which outputs a list of scores only to be used to concat to the NE embeddings
    """
    tokens = word_tokenize(text.lower())
    emotion_scores = {emotion: 0 for emotion in emotions_of_interest}
    total_words = len(tokens)
    
    if total_words == 0:
        return [0] * len(emotions_of_interest)
    
    # Calculate scores
    for token in tokens:
        if token in emotion_dict:
            for emotion in emotions_of_interest:
                emotion_scores[emotion] += emotion_dict[token][emotion]
    
    # Normalise scores
    max_scores = {emotion: max(emotion_scores.values()) for emotion in emotions_of_interest}
    normalized_scores = [emotion_scores[emotion] / (max_scores[emotion] if max_scores[emotion] > 0 else 1)
                         for emotion in emotions_of_interest]
    
    return normalized_scores


def calculate_semantic_similarity(entities_embeddings):
    if len(entities_embeddings) < 2:
        return 0.0  # Not enough entities to calculate similarity
    
    embeddings_matrix = np.array([embedding.numpy() for embedding in entities_embeddings])
    sim_matrix = cosine_similarity(embeddings_matrix)
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_similarities = sim_matrix[triu_indices]
    
    # Calculate mean of similarities
    mean_similarity = np.mean(pairwise_similarities) if len(pairwise_similarities) > 0 else 0.0
    
    return mean_similarity


def calculate_doc_ne_semantic_similarity(article_txt):
    """
    document level feature to be concatenated to the document embedding.
    this is the cosine similarity of the embeddings of the NEs found in the whole artilce. this is to understand the NE destribution in doc level
    """

    name_entities_doc_lvl = get_named_entities(article_txt)
    name_entities_doc_lvl_embeddings = [get_bert_word_embedding(word) for word in name_entities_doc_lvl]

    similarities = []
    for i in range(len(name_entities_doc_lvl_embeddings)):
        for j in range(i + 1, len(name_entities_doc_lvl_embeddings)):
            sim = 1 - cosine(name_entities_doc_lvl_embeddings[i].numpy(), name_entities_doc_lvl_embeddings[j].numpy())
            similarities.append(sim)
    
    # Compute the mean similarity
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0
    normalized_similarity = mean_similarity
    
    return normalized_similarity


def get_bert_word_embedding(topic):
    """
    Get the bert embedding for a topic.
    """
    inputs = tokenizer(topic, return_tensors='tf', truncation=True, padding=True)
    outputs = bert_model(inputs)
    hidden_states = outputs.last_hidden_state
    # Average token embeddings to get the word embedding
    return tf.reduce_mean(hidden_states[0], axis=0)


def calculate_mean_cosine_similarity_for_topics(topics:list[str]) -> float:
    """
    Calculate the mean cosine similarity of embeddings for the given list of words.
    """
    embeddings = [get_bert_word_embedding(word) for word in topics]
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = 1 - cosine(embeddings[i].numpy(), embeddings[j].numpy())
            similarities.append(sim)
    
    # Compute the mean similarity
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0
    normalized_similarity = mean_similarity
    
    return normalized_similarity


def get_segment_topics_similarity_score(seg_name:str, article:dict):
    """
    reads segment level topics from data file and measures cosine similarity between their embeddings.
    these topics must be already added using topic_analysis.py
    """
    topic1 = article[f'{seg_name}_topic_1']
    topic2 = article[f'{seg_name}_topic_2']
    topic3 = article[f'{seg_name}_topic_3']

    topics = [topic1, topic2, topic3]
    similarity_score = calculate_mean_cosine_similarity_for_topics(topics)

    return similarity_score


def get_doc_topics_similarity_score(article):
    """
    reads article level topics from data file and measures cosine similarity between their embeddings.
    these topics must be already added using topic_analysis.py
    """

    topic1 = article['Article_topic_1']
    topic2 = article['Article_topic_2']
    topic3 = article['Article_topic_3']

    topics = [topic1, topic2, topic3]
    similarity_score = calculate_mean_cosine_similarity_for_topics(topics)

    return similarity_score



def get_segment_readability_scores(seg_name:str, article:dict):
    seg_fres = article[f'{seg_name}_fres']
    seg_fkg = article[f'{seg_name}_fkg']
    seg_fog = article[f'{seg_name}_fog']

    return [seg_fres, seg_fkg, seg_fog]

def get_named_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]



if __name__ == "__main__":
    tst = "I am very tired and exhousted today"
    print(calculate_emotion_scores(tst))