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



def search_w_stentence_transformer(embedder, flat_query,
                                    corpus_list,
                                   score_ML=None, score_BM25=None,
                                   show_progress_bar=True, batch_size=8):
    """
    Compute the similarity scores of Sentence transformer.

    Parameters
    ----------
    embedder : Sentence Transformer model
        Model to use
    flat_query : string
        Query text
    corpus_list: list of string
        Texts to search in
    score_ML : list of float
        scores of the other Machine learning-based approach
    score_BM25 : list of float
        scores of the other BM25 approach
    show_progress_bar : boolean
        True to show the progress bar when computing the corpus embeddings
    batch_size : int
        batch size for Sentence Transformer inference

    Returns
    -------
    s_bert_res : list
        Results
    """

    # compute embeddings
    query_embedding = embedder.encode([flat_query])
    corpus_embeddings = embedder.encode(corpus_list, batch_size= batch_size,  show_progress_bar=show_progress_bar)

    # compute similarity
    sim_scores = cosine_similarity(query_embedding[0].reshape(1, -1),
                              np.array(corpus_embeddings))[0]

    s_bert_res = sim_scores

    # if we want to display the comparisons of methods / scores
    if (score_ML is not None) and (score_BM25 is not None):
        print('Similarity scores statistics:')
        print(scipy.stats.describe(sim_scores))

        # compute comparisons between methods
        print('Similarity between Biobert and BM25:')
        print(spearmanr(score_ML, score_BM25))
        print(kendalltau(score_ML, score_BM25))
        print('Similarity between Sentence Bert and BM25:')
        print(spearmanr(s_bert_res, score_BM25))
        print(kendalltau(s_bert_res, score_BM25))
        print('Similarity between Sentence Bert and Biobert:')
        print(spearmanr(s_bert_res, score_ML))
        print(kendalltau(s_bert_res, score_ML))

    return s_bert_res


def concat_title_abstract(ans):
    """
    Compute the title with the abstract
    to enrich the abstracts.

    Parameters
    ----------
    ans: pandas dataframe
        Dataframe with the results of BM25

    Returns
    -------
    ans: pandas dataframe
        Dataframe with the added column
    """

    # concatenate title and abstract
    title_list = list(ans['title'])
    abstract_list = list(ans['abstract'])
    ind_list = list(range(len(title_list)))
    title_abstr_list = list(map(
        lambda x: clean_hf.add_title_to_abstr(x, title_list, abstract_list),
        ind_list))
    ans['title_abstr'] = title_abstr_list

    return ans

def split_paragraph(text, min_words=2):
    """
    Split the paper text into paragraphs.

    Parameters
    ----------
    text: string
        paper text
    min_words: int
        remove paragraphs with quantity of words <= min words

    Returns
    -------
    split_text_clean: list of string
        list of paragraphs
    """

    split_text = text.split('\n\n')
    # remove last ''
    split_text = split_text[:-1]
    # remove trash
    split_text_clean = [t for t in split_text if len(t.split()) > min_words]

    return split_text_clean

def compute_parag_scores(index, parag_list, embedder, flat_query):
    """
    Compute the similarity scores of Sentence transformer.

    Parameters
    ----------
    index : int
        row index
    parag_list : list of list of string
        list of paragraphs / row
    embedder : Sentence Transformer model
        Model to use
    flat_query : string
        Query text

    Returns
    -------
    res : matrix of float
        Similarity scores / paragraphs / row
    """

    parag_paper = parag_list[index]
    res = search_w_stentence_transformer(embedder, flat_query,
                                        corpus_list=parag_paper,
                                       score_ML=None, score_BM25=None,
                                       show_progress_bar=False, batch_size=8)

    return res



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def answer_question_batch(question, answer_text, tokenizer, model, squad2=True, tau=0.5, batch_size=4):
    """
    Search the answer of a question and do the computations in batch.

    Parameters
    ----------
    question : string
        The natural language question
    answer_text : list of string
        The corpus to search in
    tokenizer : Huggingface tokenizer
        The tokenizer to use
    model : Huggingface model
        The model to use
    squad2 : boolean
        True if the model is trained on Squad2, False for Squad 1.
        In Squad 1, the model presupposes that the answer exists, and might return garbage
        if there is no answer in the text. For Squad 2, the model also predicts if there is an answer in the text
    tau : float
        For squad 2. Buffer used to decide if there is an answer.
        is_answer_found = score > cls_score + tau, see Bert paper
    batch_size : int
        the batch size

    Returns
    -------
    ans_qa_batch : Pandas dataframe
        Dataframe with the columns
    """

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
