from collections import defaultdict
import pandas as pd


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

print(emotion_dict)