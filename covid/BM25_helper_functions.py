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

from nltk.tokenize import word_tokenize
import math
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from rank_bm25 import BM25Okapi, BM25Plus # don't use BM25L, there is a mistake


nltk.download('stopwords')
nltk.download('punkt')


def get_top_n(bm25_model, query, documents, n=5):
    """
    Reimplementation of the method to get the index of the top n.

    See https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
    """

    scores = bm25_model.get_scores(query)
    top_n = np.argsort(scores)[::-1][:n]
    top_scores = scores[top_n]
    return top_n, top_scores


def clean_text_for_query_search(index, text_list, stopword_remove=True,
                                stemming=True,
                                stemmer=PorterStemmer()):
    """
    Clean-up the text to make a key-word query more efficient.

    Lower case, remove punctuation, numeric strings, stopwords,
    and finally stem.

    Note: Do not use before a language model

    Parameters
    ----------
    index : int
        Index of text in list to process, for use of vectorization
    text : String
        Text to clean
    stopword_remove : boolean
        True to remove stowords
        Article suggests that this is detrimental
    stemming : boolean
        True for stemming the text
        Article suggests that this is only useful with a very weak stemmer
    stemmer : NLTK stemmer
        Stemmer to use

    Returns
    -------
    cleaned_text : string
        Processed text
    """
    # retrieve element from list to use tqdm
    text = text_list[index]

    punc_table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))

    def process_tokens(token):
        """ Vectorize token processing."""
        token_low = token.lower()
        stripped = token_low.translate(punc_table)
        alpha = stripped if not stripped.isnumeric() else ''
        # remove stop words filtering as suggested in article
        if stopword_remove:
            alpha = alpha if not alpha in stop_words else ''
        # also disable stemmer! as suggested in article
        if stemming:
            alpha = stemmer.stem(alpha)

        return alpha

    concat_doc = list(map(lambda x: process_tokens(x), word_tokenize(text)))
    cleaned_doc = ' '.join([w for w in concat_doc if not w == ''])

    return cleaned_doc


def read_question(task_text, quest_filename):
    # get task text
    with open(quest_filename) as f:
        quest_text = task_text + f.readlines()
    return quest_text


def search_corpus_for_question(quest_text, data_df, model, top_n=10,
                               col='cleaned_text'):
    """
    Clean-up the text to make a key-word query more efficient.

    Lower case, remove punctuation, numeric strings, stopwords,
    and finally stem.

    Note: Do not use before a language model

    Parameters
    ----------
    quest_text : List of string
        Lines of the questions + task.
    data_df : Pandas dataframe
        Dataframe with corpus text
    model: BM25 model
        Model to use
    top_n: int
        quantity of results to return
    col: string
        column to search on

    Returns
    -------
    indices : list of int
        Indices of answers in the input dataframe
    scores : list of float
        scores of the top documents
    flat_query : list of string
        Prepared query text
    """
    # create BM25 model
    corpus = data_df[col]
    tokenized_corpus = [str(doc).split(" ") for doc in corpus]
    bm25 = model(tokenized_corpus)

    # prepare query
    cleaned_query = list(map(
        lambda x: clean_text_for_query_search(x, quest_text), trange(len(quest_text))))
    flat_query = " ".join(map(str, cleaned_query))
    tokenized_query = list(flat_query.split(" "))
    # search
    indices, scores = get_top_n(bm25, tokenized_query, corpus, n=top_n)

    return indices, scores, flat_query