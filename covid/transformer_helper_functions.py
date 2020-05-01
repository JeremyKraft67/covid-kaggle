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