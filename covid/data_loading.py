#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:33:49 2020

@author: jkraft
"""


import os
import json
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import nltk
import re
from time import time
from scipy.stats import spearmanr, kendalltau

from nltk.tokenize import word_tokenize
import math
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from rank_bm25 import BM25Okapi, BM25Plus # don't use BM25L, there is a mistake
import os
import torch
import numpy
from tqdm import tqdm
from transformers import *
import warnings

from covid import clean_data_helper_functions as clean_hf
from covid import BM25_helper_functions as BM25_hf

nltk.download('stopwords')
nltk.download('punkt')

# =============================================================================
# helper functions modified from 
# https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
# =============================================================================



def extract_scibert(text, tokenizer, model):
    """
    Compute the transformer embeddings.

    If the text is too long, chunk it.

    Parameters
    ----------
    text : string
        Input text
    tokenizer : Huggingface tokenizer
        Tokenizer of the model
    model : Huggingface model
        Model

    Returns
    -------
    text_ids : tensor
        Tensor of token ids of dimension 1, quantity of tokens + 2 ( for CLS, SEP)
    text_words : list of string
        List of tokens
    state : tensor
        Tensor of output embeddings of dimension 1, quantity of tokens, hidden size
    class_state : tensor
        Tensor of the CLS embedding of dimension hidden size
    layer_concat : tensor
        Tensor of the 4 last layers concatenated of dimension 1, quantity of tokens, hidden size * 4
        ordered (-4, ..., -1)
    """

    # check that the model has been configured to output the embeddings from all layers
    assert model.config.output_hidden_states == True

    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(numpy.ceil(float(text_ids.size(1)) / 510))
    states = []
    class_states = []
    layer_concats = []

    # chunk the text into passages of maximal length
    for ci in range(n_chunks):
        text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
        text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
        if text_ids[0, -1] != text_ids[0, -1]:
            text_ids_ = torch.cat([text_ids_, text_ids[0, -1].unsqueeze(0)])

        with torch.no_grad():
            res = model(text_ids_.unsqueeze(0))
            # stock all embeddings and the CLS embeddings
            state = res[0][:, 1:-1, :]
            class_state = res[0][:, 0, :]
            # compute the concatenation of the last 4 layers
            # initialize the embeddings
            all_embed = res[2]
            layer_concat = all_embed[-4][:, 1:-1, :]
            for i in range(3):
                # take only regular tokens
                layer_concat = torch.cat((layer_concat, all_embed[3 - i][:, 1:-1, :]), dim=2)

        states.append(state)
        class_states.append(class_state)
        layer_concats.append(layer_concat)

    # give back the results as tensors of dim (# tokens, # fetaures)
    state = torch.cat(states, axis=1)[0]
    class_state = torch.cat(class_states, axis=1)[0]
    layer_concat = torch.cat(layer_concats, axis=1)[0]

    return text_ids, text_words, state, class_state, layer_concat


def cross_match(state1, state2, use_CLS=False):
    if not use_CLS:
        sim = torch.cosine_similarity(torch.mean(state1, 0), torch.mean(state2, 0), dim=0)
    else:
        sim = torch.cosine_similarity(state1, state2, dim=0)
    sim = sim.numpy()
    return sim

# Prepare data
#
# # extract all data
# data_dir = '/home/jkraft/Dokumente/Kaggle/subset_data/' # 1000 documents
# all_files = clean_hf.load_files(data_dir)
# data_df = clean_hf.generate_clean_df(all_files)
#
# # prepare text for BM25
# text_to_clean = list(data_df['text'])
# index_list = list(range(len(text_to_clean)))
#
# data_df['cleaned_text'] = list(map(
#         lambda x: BM25_hf.clean_text_for_query_search(x, text_to_clean), trange(len(text_to_clean)))
#         )
#
# # prepare text for BM25
# abstract_text_to_clean = np.array(data_df['abstract'])
# # replace NaN for papers without abstracts
# is_nan_list = list(map(lambda x: str(x), list(abstract_text_to_clean)))
# abstract_text_to_clean[np.array(is_nan_list) == 'nan'] = ''
# index_list = list(range(len(abstract_text_to_clean)))
#
# data_df['cleaned_abstract'] = list(map(
#         lambda x: clean_hf.clean_text_for_query_search(
#             x, abstract_text_to_clean), trange(len(abstract_text_to_clean)))
#         )
#
# # prepare text for BM25
# abstract_text_to_clean = np.array(data_df['title'])
# # replace NaN for papers without abstracts
# is_nan_list = list(map(lambda x: str(x), list(abstract_text_to_clean)))
# abstract_text_to_clean[np.array(is_nan_list) == 'nan'] = ''
# index_list = list(range(len(abstract_text_to_clean)))
#
# data_df['cleaned_title'] = list(map(
#         lambda x: BM25_hf.clean_text_for_query_search(
#             x, abstract_text_to_clean), trange(len(abstract_text_to_clean)))
#         )
#
#
# # ind_max = 100
# # _ = list(map(
# #         lambda x: BM25_hf.clean_text_for_query_search(x, text_to_clean[:ind_max]), trange(ind_max))
# #         )
# os.chdir('/home/jkraft/Dokumente/Kaggle/')
# # save data
# data_df.to_csv("./11000_doc_df.csv", index=False)
# #

os.chdir('/home/jkraft/Dokumente/Kaggle/')

# some titles and abstract are empty in the data
data_df = pd.read_csv("./1000_doc_df.csv")

# =============================================================================
# Try BM25
# =============================================================================


task_text = ["What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?"]
quest_filename = "./question1.txt"

# quest_text = read_question(task_text, quest_filename)

# consider only papers about the coronavirus
 # don't search for the term 'coronavirus' to avoid animal diseases
quest_text = ['covid19', 'covid-19', 'cov-19', 'ncov-19', 'sarscov2', '2019novel',
              'SARS-CoV-2', '2019-nCoV', '2019nCoV', 'SARSr-CoV']
              # + ['Wuhan']

indices, scores, _ = BM25_hf.search_corpus_for_question(
    quest_text, data_df, BM25Okapi, len(data_df), 'cleaned_text')
# select only the documents containing the coronavirus terms in the abstract
# almost all documents contain the term in the text...
contain_coron_in = np.array(indices)[scores>0]
data_df_red = data_df.iloc[contain_coron_in, :].copy()
data_df_red = data_df_red.reset_index(drop=True)

# search the question, use only keywords
task_text = ['COVID-19 risk factors? epidemiological studies']
quest_text = ['COVID-19 risk factors? epidemiological studies',
 'risks factors',
 '    Smoking, pre-existing pulmonary disease',
 '    Co-infections co-existing respiratory viral infections virus transmissible virulent co-morbidities',
 '    Neonates pregnant',
 '    Socio-economic behavioral factors economic impact virus differences']


# flat_query = 'Neonates and pregnant women.'

flat_query = 'Smoking, pre-existing pulmonary disease'

flat_query = 'Socio-economic and behavioral factors to understand the economic impact of the virus'

flat_query = 'Neonates pregnant'

# flat_query = 'risks factors'
#
# flat_query = \
#     'Co-infections (co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities'

# quest_text = BM25_hf.read_question([flat_query], quest_filename)

quest_text = [flat_query]
# remove \n, but has no impact, this is already done somewhere else
model = BM25Okapi
indices, scores, quest = BM25_hf.search_corpus_for_question(
    quest_text, data_df_red, model, len(data_df_red), 'cleaned_text')
# remove again docs without keywords if searched with Okapi BM25
if model == BM25Okapi:
    contain_keyword = np.array(indices)[scores>0]
    answers = data_df_red.iloc[contain_keyword, :].copy()
    answers['scores_BM25'] = scores[scores>0]
else:
    answers = data_df_red.iloc[indices, :].copy()
    answers['scores_BM25'] = scores
answers = answers.sort_values(['scores_BM25'], ascending=False)
answers = answers.reset_index(drop=True)

ans = answers[['scores_BM25', 'title', 'abstract', 'text']]


# concatenate title and abstract
title_list = list(ans['title'])
abstract_list = list(ans['abstract'])
ind_list = list(range(len(title_list)))
title_abstr_list = list(map(
    lambda x: clean_hf.add_title_to_abstr(x, title_list, abstract_list),
    ind_list))
ans['title_abstr'] = title_abstr_list

# load model
warnings.simplefilter(action='ignore', category=FutureWarning)
tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
config = AutoConfig.from_pretrained('monologg/biobert_v1.1_pubmed', output_hidden_states=True)
model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed', config=config)
assert model.config.output_hidden_states == True

# tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
# model = AutoModelWithLMHead.from_pretrained("bert-large-cased")

# tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
# model = AutoModelWithLMHead.from_pretrained("bert-large-cased")
#
#
# # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
# # model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
# #
# tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v1")
# model = AutoModelWithLMHead.from_pretrained("albert-xxlarge-v1")
#
# tokenizer = AutoTokenizer.from_pretrained("binwang/bert-large-nli-stsb")
# model = AutoModelWithLMHead.from_pretrained("binwang/bert-large-nli-stsb")

t = time()
query_ids, query_words, query_state, query_class_state, query_layer_concat = extract_scibert(flat_query, tokenizer, model)
print((time()-t)*1000)
print(len(query_words))

# remove rows without titles / abstracts
ans_red = ans[~(ans['title_abstr'].isna() | (ans['title_abstr'] == ' '))].copy()
ans_red = ans_red.reset_index(drop=True)

use_CLS = False # TO DO: solve bug when abstract is longer than 512 tokens
use_last_four = True

# compute similarity scores
sim_scores = []
for text in tqdm(ans_red['title_abstr']):
    text_ids, text_words, state, class_state, layer_concat = extract_scibert(text, tokenizer, model)
    if use_CLS:
        sim_score = cross_match(query_class_state, class_state, True) # try cosine on CLS tokens
    elif use_last_four:
        sim_score = cross_match(query_layer_concat, layer_concat, False)
    else:
        sim_score = cross_match(query_state, state, False)
    sim_scores.append(sim_score)

# Store results in the dataframe
rel_index = np.flip(numpy.argsort(sim_scores))
end_ans = ans_red.iloc[rel_index, :]
end_ans = end_ans.reset_index(drop=True)
end_ans['score_ML'] = np.array(sim_scores)[rel_index]
# reorder columns
end_ans = end_ans[['score_ML', 'scores_BM25', 'title_abstr', 'title', 'abstract']]

end_ans_last_four = end_ans.copy()

flat_query


df = end_ans_last_four.copy()
print(spearmanr(df['score_ML'], df['scores_BM25']))
print(kendalltau(df['score_ML'], df['scores_BM25']))

# end_ans.to_csv('res_albert_xx_large.csv', index=False)

# TO DO: try more articles

# end_ans_albert = end_ans.copy() # better than biobert
# end_ans_bert_large = end_ans.copy() # not good
# end_ans_albert_text = end_ans.copy() # best
# end_ans_bert_binwang = end_ans.copy() # not good
# end_ans_bert_binwang_text = end_ans.copy() # ok-ish


