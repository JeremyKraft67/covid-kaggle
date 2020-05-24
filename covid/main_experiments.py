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
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer
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
BART = False
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
# flat_query = 'How many deaths in China?'
# flat_query = 'When has the epidemy started?'
# flat_query = 'Are the pregnant women more at risk?'
# flat_query = 'Do smoking or pre-existing pulmonary disease increase risk?'
# flat_query = 'How effective is school distancing'
#
# flat_query = 'What age group is at most risk'

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
    query_ids, query_words, query_state, query_class_state, query_layer_concat =\
        transf_hf.extract_scibert(flat_query, tokenizer_biobert, model_biobert)

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
    end_ans['score_biobert'] = np.array(sim_scores)
    # reorder columns
    end_ans = end_ans[['score_biobert'] + orig_col]
    end_ans = end_ans.sort_values(['score_biobert'], ascending=False)\
                    .reset_index(drop=True)
    ans = end_ans.copy()


if S_BERT:

    # Try sentence transformers

    search_field = 'title_abstr'

    res_col_name='score_S_Bert'
    corpus_list = list(ans[search_field])
    res = transf_hf.search_w_stentence_transformer(embedder, flat_query,
                                        corpus_list=corpus_list,
                                       show_progress_bar=True, batch_size=8)
    orig_col = list(ans.columns)
    ans[res_col_name] = res
    # reorder columns
    ans = ans[[res_col_name] + orig_col].copy()
    ans = ans.sort_values([res_col_name], ascending=False)\
                    .reset_index(drop=True)
    # print scores
    transf_hf.compare_scores(ans)


if COMPUTE_PARAGRAPH_SCORE:
    # compute paragraph scores

    orig_col = list(ans.columns)
    # compute paragraphs
    paragraph_mat = list(map(lambda x: transf_hf.split_paragraph(x), list(ans['text'])))
    # coompute scores
    res = list(map(lambda x:transf_hf.compute_parag_scores(x, paragraph_mat, embedder, flat_query),
             trange(len(ans))))
    max_parag_score = list(map(lambda x: np.max(x), res))

    # store results in dataframe
    res_col_name = 'score_max_parag'
    ans[res_col_name] = max_parag_score
    # reorder columns
    ans = ans[[res_col_name] + orig_col].copy()
    ans = ans.sort_values([res_col_name], ascending=False)\
                    .reset_index(drop=True)

    transf_hf.compare_scores(ans)


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

    # remove rows with no answers
    no_answer = (ans['answer'] == 'No answer found.') | (ans['answer'] == '')
    ans_clean = ans[~no_answer].copy()
    ans_clean = ans_clean.reset_index(drop=True)


# TRY BART


import argparse
from pathlib import Path

import torch

if BART:

    DEFAULT_DEVICE = "cpu"


    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def generate_summaries(model, tokenizer,
        df, col_name: str = 'title_abstr', batch_size: int = 8, device: str = DEFAULT_DEVICE,
        max_length_input: int = 1024):

        """
        Summarize with batch processing.

        Parameters
        ----------
        model : Huggingface model
            The model to use
        tokenizer : Huggingface tokenizer
            The tokenizer to use
        df : pandas dataframe
            The dataframe containing the paragraph to summarize
        col_name : string
            column to summarize
        batch_size : int
            the batch size
        device : str
            the device to use for running the network
        max_length_input : int
            Maximum length of input. Longer input will be truncated.

        Returns
        -------
        df : Pandas dataframe
            Input dataframe with the added columns ['summary']
        """

        examples = df[col_name]
        summ_l = []

        max_length = 100
        min_length = 30

        # choose the batches

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in list(examples)])

        # chunks
        iterator = range(0, len(examples), batch_size)
        iterator = tqdm(iterator, desc="Batches")

        # compute batches
        for batch_idx in iterator:
            # process per batch

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(examples))
            batch = length_sorted_idx[batch_start: batch_end]

            # compute the longest length in the batch
            # assume that it is for the last element

            # solve bug in indices for last element
            if batch_end != len(examples):
                longest_text = examples[length_sorted_idx[batch_end]]
            else:
                longest_text = examples[length_sorted_idx[batch_end - 1]]

            longest_seq_dct = tokenizer.batch_encode_plus([longest_text], return_tensors="pt")
            max_len = len(longest_seq_dct['input_ids'].squeeze())

            # encode th whole batch
            dct = tokenizer.batch_encode_plus(examples[batch],
                                              max_length=min(max_len, max_length_input),
                                              return_tensors="pt", pad_to_max_length=True)
            # generate batch summaries
            summaries = model.generate(
                input_ids=dct["input_ids"].to(device),
                attention_mask=dct["attention_mask"].to(device),
                num_beams=5,
                temperature=1,
                length_penalty=1.0,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=model.config.eos_token_id,
            )
            summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

            # store the results
            summ_l.append(summ)

        # restore order

        flat_summ_l = [item for sublist in summ_l for item in sublist]

        # create dataframe results
        summary_batch = pd.DataFrame(zip(flat_summ_l, length_sorted_idx),
                                    columns=['summary', 'original_idx'])
        summary_batch['original_idx'] = summary_batch['original_idx'].astype(int)
        summary_batch = summary_batch.sort_values(['original_idx'], ascending=True) \
            .reset_index(drop=True)

        # copy results in input dataframe
        df['summary'] = summary_batch['summary']

        return df

    model_name= "bart-large-xsum"

    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    res = generate_summaries(model, tokenizer,
        df=ans_clean, batch_size = 4, device= DEFAULT_DEVICE)


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

# def generate_summaries(model, tokenizer,
#     examples: list, batch_size: int = 8, device: str = DEFAULT_DEVICE,
#     max_length_input: int = 512):
#
#     max_length = 100
#     min_length = 30
#
#     for batch in tqdm(list(chunks(examples, batch_size))):
#         dct = tokenizer.batch_encode_plus(batch, max_length=max_length_input, return_tensors="pt", pad_to_max_length=True)
#
#         # tokenizer.decode(
#         #     dct['input_ids'].squeeze())
#
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
