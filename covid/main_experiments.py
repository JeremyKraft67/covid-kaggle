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
from tqdm import tqdm, trange
from transformers import *
import warnings

from covid import clean_data_helper_functions as clean_hf
from covid import BM25_helper_functions as BM25_hf
from covid import transformer_helper_functions as transf_hf

# import clean_data_helper_functions as clean_hf
# import BM25_helper_functions as BM25_hf

nltk.download('stopwords')
nltk.download('punkt')


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


BIOBERT = False
S_BERT = False
COMPUTE_PARAGRAPH_SCORE = False
QA = True
top_res = 50

# TO DO: Remove the print of column score for columns that were not computed

# prepare models

if BIOBERT:
    # load model
    tokenizer_biobert = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
    config = AutoConfig.from_pretrained('monologg/biobert_v1.1_pubmed', output_hidden_states=True)
    model_biobert = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed', config=config)
    assert model_biobert.config.output_hidden_states == True
if S_BERT:
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
if QA:
    # QA
    # tokenizer = AutoTokenizer.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")
    # model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")
    tokenizer_qa = AutoTokenizer.from_pretrained("clagator/biobert_squad2_cased")
    model_qa = AutoModelForQuestionAnswering.from_pretrained("clagator/biobert_squad2_cased")
    # tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    # model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    # tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
    # model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")


# flat_query = 'Smoking, pre-existing pulmonary disease'

# flat_query = 'Socio-economic and behavioral factors to understand the economic impact of the virus'
#
# flat_query = 'Neonates pregnant'

flat_query = 'risks factors'
flat_query = 'What is the incubation period?'
flat_query = 'Are there mutations?'
flat_query = 'How many deaths in China?'
flat_query = 'When has the epidemy started?'
# flat_query = 'incubation period'

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
ans = transf_hf.concat_title_abstract(ans)
# clean-up
# remove rows without titles / abstracts
ans = ans[~(ans['title_abstr'].isna() | (ans['title_abstr'] == ' '))].iloc[:top_res, :].copy()
ans = ans.reset_index(drop=True)


if BIOBERT:

    # BERT embeddings similarity

    # consider only top results from BM25
    use_CLS = False # TO DO: solve bug when abstract is longer than 512 tokens
    use_last_four = True
    search_field = 'title_abstr'

    # process query
    t = time()
    query_ids, query_words, query_state, query_class_state, query_layer_concat =\
        transf_hf.extract_scibert(flat_query, tokenizer_biobert, model_biobert)
    print((time()-t)*1000)
    print(len(query_words))

    # compute similarity scores
    sim_scores = []
    for text in tqdm(ans[search_field]):
        text_ids, text_words, state, class_state, layer_concat = transf_hf.extract_scibert(text, tokenizer_biobert, model_biobert)
        if use_CLS:
            sim_score = transf_hf.cross_match(query_class_state, class_state, True) # try cosine on CLS tokens
        elif use_last_four:
            sim_score = transf_hf.cross_match(query_layer_concat, layer_concat, False)
        else:
            sim_score = transf_hf.cross_match(query_state, state, False)
        sim_scores.append(sim_score)

    # Store results in the dataframe
    end_ans = ans.copy()
    orig_col = list(ans.columns)
    end_ans['score_ML'] = np.array(sim_scores)
    # reorder columns
    end_ans = end_ans[['score_ML'] + orig_col]
    end_ans = end_ans.sort_values(['score_ML'], ascending=False)\
                    .reset_index(drop=True)
    ans = end_ans.copy()

if S_BERT:

    # Try sentence transformers

    search_field = 'title_abstr'

    res_col_name='score_S_Bert'
    corpus_list = list(ans[search_field])
    score_ML = ans['score_ML']
    score_BM25 = ans['scores_BM25']

    res = transf_hf.search_w_stentence_transformer(embedder, flat_query,
                                        corpus_list=corpus_list,
                                       score_ML=score_ML, score_BM25=score_BM25,
                                       show_progress_bar=True, batch_size=8)
    orig_col = list(ans.columns)
    ans[res_col_name] = res
    # reorder columns
    ans = ans[[res_col_name] + orig_col].copy()
    ans = ans.sort_values([res_col_name], ascending=False)\
                    .reset_index(drop=True)

if COMPUTE_PARAGRAPH_SCORE:
    # compute paragraph scores

    orig_col = list(ans.columns)
    # compute paragraphs
    paragraph_mat = list(map(lambda x: transf_hf.split_paragraph(x), list(ans['text'])))
    # coompute scores
    res = list(map(lambda x:transf_hf. compute_parag_scores(x, paragraph_mat, embedder, flat_query),
             trange(len(ans))))
    max_parag_score = list(map(lambda x: np.max(x), res))

    # print results
    print(scipy.stats.describe(max_parag_score))
    score_ML = ans['score_ML']
    score_s_bert = ans['score_S_Bert']
    score_BM25 = ans['scores_BM25']
    # compute comparisons between methods
    print('Similarity between Biobert and BM25:')
    print(spearmanr(score_ML, score_BM25))
    print(kendalltau(score_ML, score_BM25))
    print('Similarity between Sentence Bert and BM25:')
    print(spearmanr(score_s_bert, score_BM25))
    print(kendalltau(score_s_bert, score_BM25))
    print('Similarity between max S-bert paragraph scores and BM25:')
    print(spearmanr(max_parag_score, score_BM25))
    print(kendalltau(max_parag_score, score_BM25))
    print('Similarity between max S-bert paragraph + S-Bert scores and BM25:')
    print(spearmanr(max_parag_score + score_s_bert, score_BM25))
    print(kendalltau(max_parag_score + score_s_bert, score_BM25))
    # store results in dataframe
    res_col_name = 'score_max_parag'
    ans[res_col_name] = max_parag_score
    # reorder columns
    ans = ans[[res_col_name] + orig_col].copy()
    ans = ans.sort_values([res_col_name], ascending=False)\
                    .reset_index(drop=True)


# TRY QA

if QA:

    # base_quest = 'Risk factors for '
    # group = 'neonates and pregnant women?'
    # question = base_quest + group
    #
    # question = 'What are the risk factors?'

    # search articles

    # compute answer scores

    answer_text = list(ans['title_abstr'])
    ans_qa_batch = transf_hf.answer_question_batch(flat_query, answer_text, tokenizer_qa, model_qa, squad2=True, tau=5, batch_size=4)

    # merge qa results
    # drop qa answer columns to do another search
    if 'score_qa' in ans.columns:
        ans = ans.drop(['score_qa', 'answer', 'original_idx'], axis=1)
    orig_col = list(ans.columns)

    ans['original_idx'] = list(range(len(ans)))
    ans = ans.merge(ans_qa_batch, on='original_idx')
    ans = ans[['score_qa', 'answer', 'original_idx'] + orig_col]
    ans = ans.sort_values(['score_qa'], ascending=False) \
        .reset_index(drop=True)




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

#
#
# # TRY BART
#
#
# import argparse
# from pathlib import Path
#
# import torch
# from tqdm import tqdm
#
# from transformers import BartForConditionalGeneration, BartTokenizer
#
#
# DEFAULT_DEVICE = "cpu"
#
#
# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]
#
#
# def generate_summaries(
#     examples: list,  model_name: str = "bart-large-cnn", batch_size: int = 8, device: str = DEFAULT_DEVICE
# ):
#
#     model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
#     tokenizer = BartTokenizer.from_pretrained(model_name)
#
#     max_length = 100
#     min_length = 30
#
#     for batch in tqdm(list(chunks(examples, batch_size))):
#         dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
#         t = time()
#         summaries = model.generate(
#             input_ids=dct["input_ids"].to(device),
#             attention_mask=dct["attention_mask"].to(device),
#             num_beams=5,
#             temperature=1,
#             length_penalty=1.0,
#             max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
#             min_length=min_length + 1,  # +1 from original because we start at step=1
#             no_repeat_ngram_size=3,
#             early_stopping=True,
#             decoder_start_token_id=model.config.eos_token_id,
#         )
#         print((time() - t) * 1000)
#         dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
#         print((time() - t) * 1000)
#
#         return dec
#
#
# model_name= "bart-large-xsum"
# model_name= "bart-large-cnn"
#
# res = generate_summaries(
#     examples=[ans['title_abstr'][1]], model_name= model_name, batch_size = 8, device= DEFAULT_DEVICE)
#
# print(res)
#
# len(ans['title_abstr'][1].split())
#
# model= "bart-large-xsum"
# tokenizer = AutoTokenizer.from_pretrained("bart-large-xsum")
# BartTokenizer.from_pretrained("bart-large-xsum")
# model = AutoModelWithLMHead.from_pretrained("bart-large-xsum")
