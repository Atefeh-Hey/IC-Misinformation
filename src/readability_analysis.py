"""
run this to add readability scores at segment level to the data
"""

import pandas as pd

from utils import calculate_readability_score


#  ###     SELECT THE TARGET SEGMENT TO EXTRACT READABILITY FOR IT....    ###
target_segment = 'Title'
# target_segment = 'FP'
# target_segment = 'Body'

df = pd.read_csv('data.csv')


for index, row in df.iterrows():
    segment_txt = row[target_segment]

    if segment_txt:
        readability_scores = calculate_readability_score(segment_txt)
    else:
        readability_scores = [0, 0, 0]

    df.at[index, f'{target_segment}_fres'] = readability_scores[0]
    df.at[index, f'{target_segment}_fkg'] = readability_scores[1]
    df.at[index, f'{target_segment}_fog'] = readability_scores[2]





