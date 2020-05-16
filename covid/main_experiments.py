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


flat_query = 'Incubation period.'

# flat_query = 'Smoking, pre-existing pulmonary disease'

# flat_query = 'Socio-economic and behavioral factors to understand the economic impact of the virus'
#
# flat_query = 'Neonates pregnant'

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
ans = transf_hf.concat_title_abstract(ans)

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

search_field = 'title_abstr'

embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

res_col_name='score_S_Bert'
corpus_list = list(end_ans[search_field])
score_ML = end_ans['score_ML']
score_BM25 = end_ans['scores_BM25']

res = transf_hf.search_w_stentence_transformer(embedder, flat_query,
                                    corpus_list=corpus_list,
                                   score_ML=score_ML, score_BM25=score_BM25,
                                   show_progress_bar=True, batch_size=8)
end_ans_s_bert = end_ans.copy()
end_ans_s_bert[res_col_name] = res
# reorder columns
end_ans_s_bert = end_ans_s_bert[[res_col_name, 'score_ML', 'scores_BM25', 'title_abstr', 'title', 'abstract', 'text']].copy()
end_ans_s_bert = end_ans_s_bert.sort_values([res_col_name], ascending=False)\
                .reset_index(drop=True)

# compute paragraph scores

# compute paragraphs
paragraph_mat = list(map(lambda x: transf_hf.split_paragraph(x), list(end_ans_s_bert['text'])))
end_ans_s_bert['paragraphs'] = paragraph_mat
# coompute scores
parag_list = end_ans_s_bert['paragraphs']
res = list(map(lambda x:transf_hf. compute_parag_scores(x, parag_list, embedder, flat_query),
         trange(len(end_ans_s_bert))))
max_parag_score = list(map(lambda x: np.max(x), res))

# print results
print(scipy.stats.describe(max_parag_score))
score_ML = end_ans_s_bert['score_ML']
score_s_bert = end_ans_s_bert['score_S_Bert']
score_BM25 = end_ans_s_bert['scores_BM25']
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
end_ans_s_bert[res_col_name] = max_parag_score
# reorder columns
end_ans_s_bert = end_ans_s_bert[[res_col_name, 'score_S_Bert', 'score_ML', 'scores_BM25', 'title_abstr', 'title',
       'abstract', 'text', 'paragraphs']].copy()
end_ans_s_bert = end_ans_s_bert.sort_values([res_col_name], ascending=False)\
                .reset_index(drop=True)

# test = end_ans_s_bert['paragraphs']

# # compute stats of paragraph lengths
# # only a fraction of the paragraphs should be trimmed by Bert
# para_len = list(map(lambda x: [len(para.split()) for para in x] , list(paragraph_mat)))
# flat_len_list = [item for sublist in para_len for item in sublist]
# print(scipy.stats.describe(flat_len_list))
# percent = range(0, 100, 2)
# print(np.percentile(flat_len_list, percent))




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
#     examples=[end_ans_s_bert['title_abstr'][1]], model_name= model_name, batch_size = 8, device= DEFAULT_DEVICE)
#
# print(res)
#
# len(end_ans_s_bert['title_abstr'][1].split())
#
# model= "bart-large-xsum"
# tokenizer = AutoTokenizer.from_pretrained("bart-large-xsum")
# BartTokenizer.from_pretrained("bart-large-xsum")
# model = AutoModelWithLMHead.from_pretrained("bart-large-xsum")


# TRY QA





# tokenizer = AutoTokenizer.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")
# model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")
#
tokenizer = AutoTokenizer.from_pretrained("clagator/biobert_squad2_cased")
model = AutoModelForQuestionAnswering.from_pretrained("clagator/biobert_squad2_cased")
#
# tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")


def answer_question(question, answer_text, tokenizer, model, squad2=True, tau=0.5):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    token_dict = tokenizer.encode_plus(question, answer_text, max_length = model.config.max_position_embeddings)



    # compute scores (as before the softmax)
    start_scores, end_scores = model(torch.tensor([token_dict['input_ids']]),
                                     token_type_ids=torch.tensor([token_dict['token_type_ids']]))


    # get the best start and end score, with start <= end

    # compute the upper triangular matrix of Mij = start_i + end_j
    start_scores = start_scores.data.numpy().flatten()
    end_scores = end_scores.data.numpy().flatten()

    mat_score = np.tile(start_scores, (len(start_scores), 1)).transpose()
    mat_score = mat_score + np.tile(end_scores, (len(end_scores), 1))
    # take the upper triangular matrix to make sure that the end index >= start index
    mat_score = np.triu(mat_score)

    # find the indices of the maximum
    arg_max_ind = np.argmax(mat_score.flatten())
    answer_start = arg_max_ind // len(mat_score)
    answer_end = arg_max_ind % len(mat_score)
    assert np.max(mat_score) == mat_score[answer_start, answer_end]
    score = mat_score.flatten()[arg_max_ind]

    # check if answer is found (score > CLS_score + tau, see paper)
    # otherwise return no answer
    if squad2:
        # check if answer exists
        cls_score = start_scores[0] + end_scores[0]
        is_answer_found = score > cls_score + tau
        # redefine answer
        score = score if is_answer_found else cls_score
        answer_start = answer_start if is_answer_found else 0
        answer_end = answer_end if is_answer_found else 0

    answer = tokenizer.decode(token_dict['input_ids'][int(answer_start): int(answer_end)+1], skip_special_tokens = False, clean_up_tokenization_spaces = True)
    answer = answer if answer != '[CLS]' else 'No answer found.'

    return answer, score


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def answer_question_batch(question, answer_text, tokenizer, model, squad2=True, tau=0.5, batch_size=4):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.

    answer_l = []
    score_l = []

    all_embeddings = []
    length_sorted_idx = np.argsort([len(sen) for sen in answer_text])

    # chunks
    iterator = range(0, len(answer_text), batch_size)
    iterator = tqdm(iterator, desc="Batches")

    # compute batches
    for batch_idx in iterator:
        # process per batch
        start_scores_l = []
        end_scores_l = []

        batch_start = batch_idx
        batch_end = min(batch_start + batch_size, len(answer_text))
        batch = length_sorted_idx[batch_start: batch_end]

        # compute the longest length in the batch
        # assume that it is for the last element

        # solve bug in indices for last element
        if batch_end != len(answer_text):
            longest_text = answer_text[length_sorted_idx[batch_end]]
        else:
            longest_text = answer_text[length_sorted_idx[batch_end - 1]]
        longest_seq = tokenizer.encode_plus(question,
                              answer_text[length_sorted_idx[batch_end-1]],
                              max_length=model.config.max_position_embeddings,
                              pad_to_max_length=False,
                              return_attention_masks=True)['input_ids']
        max_len = len(longest_seq)


        token_dict_batch = [tokenizer.encode_plus(question,
                                                  answer_text[text_id],
                                                  max_length = min(max_len, model.config.max_position_embeddings),
                                                  pad_to_max_length=True,
                                                  return_attention_masks=True)
                            for text_id in batch]

        input_ids = torch.tensor([token_dict_batch[i]['input_ids'] for i in range(len(token_dict_batch))])
        token_ids = torch.tensor([token_dict_batch[i]['token_type_ids'] for i in range(len(token_dict_batch))])
        attention_mask = torch.tensor([token_dict_batch[i]['attention_mask'] for i in range(len(token_dict_batch))])

        # compute scores (as before the softmax)
        start_scores, end_scores = model(input_ids,
                                         token_type_ids=token_ids,
                                         attention_mask=attention_mask)

        # save scores
        start_scores = start_scores.data.numpy()
        end_scores = end_scores.data.numpy()

        start_scores_l.append(start_scores)
        end_scores_l.append(end_scores)

        # reconstruct the answers
        for i in range(len(start_scores_l[0])):
            start_scores = start_scores_l[0][i]
            end_scores = end_scores_l[0][i]

            # get the best start and end score, with start <= end

            # compute the upper triangular matrix of Mij = start_i + end_j

            mat_score = np.tile(start_scores, (len(start_scores), 1)).transpose()
            mat_score = mat_score + np.tile(end_scores, (len(end_scores), 1))
            # take the upper triangular matrix to make sure that the end index >= start index
            mat_score = np.triu(mat_score)

            # find the indices of the maximum
            arg_max_ind = np.argmax(mat_score.flatten())
            answer_start = arg_max_ind // len(mat_score)
            answer_end = arg_max_ind % len(mat_score)
            assert np.max(mat_score) == mat_score[answer_start, answer_end]
            score = mat_score.flatten()[arg_max_ind]

            # check if answer is found (score > CLS_score + tau, see paper)
            # otherwise return no answer
            if squad2:
                # check if answer exists
                cls_score = start_scores[0] + end_scores[0]
                is_answer_found = score > cls_score + tau
                # redefine answer
                score = score if is_answer_found else cls_score
                answer_start = answer_start if is_answer_found else 0
                answer_end = answer_end if is_answer_found else 0
            # stock the answer
            answer = tokenizer.decode(
                token_dict_batch[i]['input_ids'][int(answer_start): int(answer_end)+1],
                skip_special_tokens = False,
                clean_up_tokenization_spaces = True)
            answer = answer if answer != '[CLS]' else 'No answer found.'
            answer_l.append(answer)
            score_l.append(score)


    # create dataframe results
    ans_qa_batch = pd.DataFrame(zip(score_l, answer_l, length_sorted_idx), columns=['score_qa', 'answer', 'original_idx'])
    ans_qa_batch['original_idx'] = ans_qa_batch['original_idx'].astype(int)
    ans_qa_batch = ans_qa_batch.sort_values(['score_qa'], ascending=False) \
        .reset_index(drop=True)

    return ans_qa_batch


answer_text = 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).'

question = "What is Bert?"

question = "How long does it take to download?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

answer_text = end_ans_s_bert['title_abstr'][0]

base_quest = 'What are the risk factors for '

base_quest = 'Risk factors for '
group = 'neonates and pregnant women?'
question = base_quest + group

question = 'What are the risk factors?'

question = 'What is the incubation period?'

t = time()
answer, score = answer_question(question, answer_text, tokenizer, model)
print((time()-t)*1000)

# search articles

# compute answer scores
ans_scores_list = []
answer_list = []
for text in tqdm(end_ans_s_bert['title_abstr']):
    answer, score = answer_question(question, text, tokenizer, model)
    answer_list.append(answer)
    ans_scores_list.append(score)

ans_qa = pd.DataFrame(zip(ans_scores_list, answer_list), columns=['score', 'answer'])
ans_qa = ans_qa.sort_values(['score'], ascending=False)\
                .reset_index(drop=True)

# end_ans_s_bert.to_csv('end_ans_s_bert.csv', index=False)

answer_text = list(end_ans_s_bert['title_abstr'])

ans_qa_batch = answer_question_batch(question, answer_text, tokenizer, model, squad2=True, tau=5, batch_size=4)

# merge qa results
end_ans_s_bert['original_idx'] = list(range(len(end_ans_s_bert)))

end_ans_s_bert = end_ans_s_bert.merge(ans_qa_batch, on='original_idx')
end_ans_s_bert = end_ans_s_bert[['score_qa', 'answer', 'score_max_parag', 'score_S_Bert', 'score_ML', 'scores_BM25',
       'title_abstr', 'title', 'abstract', 'text', 'paragraphs']]

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