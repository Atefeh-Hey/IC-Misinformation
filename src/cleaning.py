import pandas as pd
import numpy as np
import langid
import nltk

def clean_eng_n_empty():
    csv = "/Users/atefeh/Desktop/fake news data/extracted/data_raw.csv"
    df = pd.read_csv(csv)

    dropped_empty_content = 0  
    dropped_non_english_content = 0  
    dropped_non_english_title = 0  

    for index, row in df.iterrows():
        text = row['content']

        if pd.isna(text) or text.strip() == "":
            print(f'EMPTY content at index: {index}')
            df.drop(index, inplace=True)
            dropped_empty_content += 1 
            continue

        # classify language for content and title
        lang_content, _ = langid.classify(text)
        lang_title, _ = langid.classify(row['Title'])

        # Drop if content is not in Eng
        if lang_content != "en":
            df.drop(index, inplace=True)
            dropped_non_english_content += 1
            continue

        # Drop if title is not in English
        if lang_title != "en":
            df.drop(index, inplace=True)
            dropped_non_english_title += 1

    print(f"dropped for empty content: {dropped_empty_content}")
    print(f"dropped for non-Eng content: {dropped_non_english_content}")
    print(f"dropped for non-Eng title: {dropped_non_english_title}")
    df.to_csv(csv, index=False)


def clean_less_than_10_sent():
    csv = "/Users/atefeh/Desktop/fake news data/extracted/data_raw.csv"
    df = pd.read_csv(csv)

    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    removed_count = 0

    for index, row in df.iterrows():
        content = row.get('content', '')
        sentences = tokenizer.tokenize(content)

        if len(sentences) < 10:
            df.drop(index, inplace=True)
            removed_count += 1  # Increment counter for removed samples

    print(f'deleted fo less than 10 sentences: {removed_count}')

    df.to_csv(csv, index=False)


if __name__ == "__main__":
    # clean_eng_n_empty()
    # clean_less_than_10_sent()