
import pandas as pd
import os
from tqdm import tqdm
import json



def format_name(author):
    middle_name = " ".join(author['middle'])
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])

def format_authors(authors):
    name_ls = []
    for author in authors:
        name = format_name(author)
        name_ls.append(name)
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    for section, text in texts:
        texts_di[section] += text
    body = ""
    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    return body

def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []
    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_body(file['abstract']),
            format_body(file['body_text']),
        ]
        cleaned_files.append(features)
    col_names = ['paper_id', 'title', 'authors',
                 'abstract', 'text']
    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    return clean_df
