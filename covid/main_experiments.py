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
from sentence_transformers import SentenceTransformer
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi, BM25Plus # don't use BM25L, there is a mistake
import os
import torch
import numpy
from tqdm import tqdm
from transformers import *
import warnings

from covid import clean_data_helper_functions as clean_hf
from covid import BM25_helper_functions as BM25_hf
from covid import transformer_helper_functions as transf_hf

# import clean_data_helper_functions as clean_hf
# import BM25_helper_functions as BM25_hf

nltk.download('stopwords')
nltk.download('punkt')

def concat_title_abstract(ans):
    # concatenate title and abstract
    title_list = list(ans['title'])
    abstract_list = list(ans['abstract'])
    ind_list = list(range(len(title_list)))
    title_abstr_list = list(map(
        lambda x: clean_hf.add_title_to_abstr(x, title_list, abstract_list),
        ind_list))
    ans['title_abstr'] = title_abstr_list

    return ans

os.chdir('/home/jkraft/Dokumente/Kaggle/')

# load prepared 50.000 papers, and filtered to contain covid 19 keywords, about ~ 1900 papers remaining
data_df_red = pd.read_csv("./all_doc_df.csv")

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

ans = answers[['scores_BM25', 'title', 'abstract', 'text']].copy()
ans = concat_title_abstract(ans)

# BERT embeddings similarity

# consider only top results from BM25
top_res = 50
use_CLS = False # TO DO: solve bug when abstract is longer than 512 tokens
use_last_four = True
search_field = 'title_abstr'

# remove rows without titles / abstracts
ans_red = ans[~(ans['title_abstr'].isna() | (ans['title_abstr'] == ' '))].iloc[:top_res, :].copy()
ans_red = ans_red.reset_index(drop=True)

# load model
warnings.simplefilter(action='ignore', category=FutureWarning)
tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
config = AutoConfig.from_pretrained('monologg/biobert_v1.1_pubmed', output_hidden_states=True)
model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed', config=config)
assert model.config.output_hidden_states == True

# process query
t = time()
query_ids, query_words, query_state, query_class_state, query_layer_concat =\
    transf_hf.extract_scibert(flat_query, tokenizer, model)
print((time()-t)*1000)
print(len(query_words))

# compute similarity scores
sim_scores = []
for text in tqdm(ans_red[search_field]):
    text_ids, text_words, state, class_state, layer_concat = transf_hf.extract_scibert(text, tokenizer, model)
    if use_CLS:
        sim_score = transf_hf.cross_match(query_class_state, class_state, True) # try cosine on CLS tokens
    elif use_last_four:
        sim_score = transf_hf.cross_match(query_layer_concat, layer_concat, False)
    else:
        sim_score = transf_hf.cross_match(query_state, state, False)
    sim_scores.append(sim_score)

# Store results in the dataframe
end_ans = ans_red.copy()
end_ans['score_ML'] = np.array(sim_scores)
# reorder columns
end_ans = end_ans[['score_ML', 'scores_BM25', 'title_abstr', 'title', 'abstract', 'text']]
end_ans = end_ans.sort_values(['score_ML'], ascending=False)\
                .reset_index(drop=True)


# Try sentence transformers

search_field = 'text'

embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
query_embedding = embedder.encode([flat_query])

t = time()
corpus_embeddings = embedder.encode(list(end_ans[search_field]), batch_size= 8,  show_progress_bar=True)
print((time()-t)*1000)

# compute similarity
sim_scores = []
for query in tqdm(corpus_embeddings):
    sim_scores.append(
        cosine_similarity(query_embedding[0].reshape(1, -1),
                          query.reshape(1, -1))[0][0])

print(scipy.stats.describe(sim_scores))

# Store results in the dataframe
end_ans_s_bert = end_ans.copy()
end_ans_s_bert['score_S_Bert'] = np.array(sim_scores)
# reorder columns
end_ans_s_bert = end_ans_s_bert[['score_S_Bert', 'score_ML', 'scores_BM25', 'title_abstr', 'title', 'abstract', 'text']]
end_ans_s_bert = end_ans_s_bert.sort_values(['score_S_Bert'], ascending=False)\
                .reset_index(drop=True)

# df = end_ans_s_bert.iloc[:20,].copy()

# compute comparisons between methods
df = end_ans_s_bert.copy()
print(spearmanr(df['score_ML'], df['scores_BM25']))
print(kendalltau(df['score_ML'], df['scores_BM25']))

print(spearmanr(df['score_S_Bert'], df['scores_BM25']))
print(kendalltau(df['score_S_Bert'], df['scores_BM25']))

print(spearmanr(df['score_S_Bert'], df['score_ML']))
print(kendalltau(df['score_S_Bert'], df['score_ML']))

# end_ans.to_csv('res_albert_xx_large.csv', index=False)

# TO DO: try more articles

# end_ans_albert = end_ans.copy() # better than biobert
# end_ans_bert_large = end_ans.copy() # not good
# end_ans_albert_text = end_ans.copy() # best
# end_ans_bert_binwang = end_ans.copy() # not good
# end_ans_bert_binwang_text = end_ans.copy() # ok-ish
# sentence bert = end_ans.copy() # better than biobert and ~ *3 faster without batching in biobert

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