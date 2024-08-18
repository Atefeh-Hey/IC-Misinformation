"""
use this to annotate data file with segment level topics for title, body and fp. 
also can select target_segment = 'Article' to calculate and add doc level topics.
set the target_segment to generate the topics for the target segment.
textrazor api keys are required for this. each api key can annotate 500 samples per day for free.
otherwise a paied account is required
"""
import json
import os
from dotenv import load_dotenv
import textrazor
import pandas as pd
import matplotlib.pyplot as plt

#  ###     SELECT THE TARGET SEGMENT TO EXTRACT TOPIC FOR IT....    ###
target_segment = 'Article'
target_segment = 'Title'
# target_segment = 'FP'
# target_segment = 'Body'

load_dotenv()

# USE ONE KEY FOR 500 SAMPLES THEN CHANGE TO ANOTHER KEY DUE TO THE LIMITATION OF THE FREE TEXTRAZOR API
# textrazor.apikey = os.getenv("TEXTRAZOR_APIKEY_1")
# textrazor.apikey = os.getenv("TEXTRAZOR_APIKEY_2")
# textrazor.apikey = os.getenv("TEXTRAZOR_APIKEY_3")
textrazor.apikey = os.getenv("TEXTRAZOR_APIKEY_4")
# textrazor.apikey = os.getenv("TEXTRAZOR_APIKEY_5")
# textrazor.apikey = os.getenv("TEXTRAZOR_APIKEY_6")

# df = pd.read_csv('/Users/atefeh/Documents/data_df_news_clean600_EMO_SENT_RB_TOP.csv')
df = pd.read_csv('data.csv')

#
client = textrazor.TextRazor(extractors=["topics"])
client.set_classifiers(["textrazor_newscodes"])
#
for index, row in df.iterrows():

    # if index < 500:
    #     continue
    if index == 500:
        break
    print(index)

    response = client.analyze(row[target_segment])
    json_content = response.json
    content = json.dumps(json_content)

    if 'coarseTopics' in json_content:
        topics_size = len(json_content['response']['coarseTopics'])
        if topics_size > 2:
            df.at[index, f'{target_segment}_topic_1'] = json_content['response']['coarseTopics'][0]['label']
            df.at[index, f'{target_segment}_topic_2'] = json_content['response']['coarseTopics'][1]['label']
            df.at[index, f'{target_segment}_topic_3'] = json_content['response']['coarseTopics'][2]['label']
        elif topics_size == 2:
            df.at[index, f'{target_segment}_topic_1'] = json_content['response']['coarseTopics'][0]['label']
            df.at[index, f'{target_segment}_topic_2'] = json_content['response']['coarseTopics'][1]['label']
            df.at[index, f'{target_segment}_topic_3'] = 'Other'
        elif topics_size == 1:
            df.at[index, f'{target_segment}_topic_1'] = json_content['response']['coarseTopics'][0]['label']
            df.at[index, f'{target_segment}_topic_2'] = 'Other'
            df.at[index, f'{target_segment}_topic_3'] = 'Other'
    else:
        df.at[index, f'{target_segment}_topic_1'] = 'Other'
        df.at[index, f'{target_segment}_topic_2'] = 'Other'
        df.at[index, f'{target_segment}_topic_3'] = 'Other'

# df.to_csv('/Users/atefeh/Documents/data_df_news_clean600_EMO_SENT_RB_TOP.csv')
df.to_csv('data.csv')