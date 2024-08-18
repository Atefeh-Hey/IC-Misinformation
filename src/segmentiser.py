"""
run this to separate each article in the dataset into Title, FP, and Body. these are the segments.

"""

import pandas as pd
import nltk.data
import nltk
import en_core_web_sm       # download the model using --> python -m spacy download en_core_web_sm

nlp = en_core_web_sm.load()

tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


# separate the 5 first paragraphs of the 'text' column and put them in a new column named 'first_parag' to
# facilitate further processing. saves it in file.
def segmentise_the_article(df_data):
    no_rows = len(df_data.index)
    for index, row in df_data.iterrows():

        # split the 'text' column into sentences
        sentences = tokenizer.tokenize(row['Cleaned'])

        # remove shorter than 10 sentences news
        # print(len(sentences))
        if len(sentences) < 10:
            df_data.drop(index, inplace=True)
            print('id ====', row[0])
        else:
            # select the first 5 sentences and put them in 'first_parag' column
            First_sentences = sentences[:5]
            FP = " ".join(First_sentences)
            df_data.at[index, 'FP'] = FP

            # put the rest of the sentences in 'text' column
            other_sentences = sentences[5:]
            body = " ".join(other_sentences)
            df_data.at[index, 'body'] = body

    no_rows = no_rows - len(df_data.index)
    print('Number of dropped rows = ', no_rows)
    print('Number of remaining rows = ', len(df_data.index))
    # df_data.reset_index()

    # new_cols_order = ['ID', 'title', 'first_parag', 'body', 'text', 'label', 'label_coh']
    # new_cols_order = ['ID', 'title', 'FP', 'body', 'text', 'label']    # when no coh label added!
    # df_data = df_data.loc[:, new_cols_order]
    return df_data


df = pd.read_csv('data.csv')

segmentised_df = segmentise_the_article(df)

segmentised_df.to_csv('data.csv', index=False)
