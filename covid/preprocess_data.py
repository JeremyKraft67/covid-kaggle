
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
from covid import transformer_helper_functions as transf_hf

# import clean_data_helper_functions as clean_hf
# import BM25_helper_functions as BM25_hf

nltk.download('stopwords')
nltk.download('punkt')

# =============================================================================
# helper functions modified from
# https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
# =============================================================================


# # Prepare data
#
# # extract all data
# data_dir = '/home/jkraft/Dokumente/Kaggle/all_data/' # 1000 documents
# all_files = clean_hf.load_files(data_dir)
# data_df = clean_hf.generate_clean_df(all_files)
#
# # reduce the size of the dataframe
# # data_df = data_df.iloc[:10,:]
# ł
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
#         lambda x: BM25_hf.clean_text_for_query_search(
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
# data_df.to_csv("./all_doc_df.csv", index=False)
# #

os.chdir('/home/jkraft/Dokumente/Kaggle/')

# # some titles and abstract are empty in the data
# data_df = pd.read_csv("./all_doc_df.csv")

# =============================================================================
# Try BM25
# =============================================================================


# task_text = ["What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?"]
# quest_filename = "./question1.txt"
#
# # quest_text = read_question(task_text, quest_filename)
#
# # consider only papers about the coronavirus
#  # don't search for the term 'coronavirus' to avoid animal diseases
# quest_text = ['covid19', 'covid-19', 'cov-19', 'ncov-19', 'sarscov2', '2019novel',
#               'SARS-CoV-2', '2019-nCoV', '2019nCoV', 'SARSr-CoV']
#               # + ['Wuhan']
#
# indices, scores, _ = BM25_hf.search_corpus_for_question(
#     quest_text, data_df, BM25Okapi, len(data_df), 'cleaned_text')
# # select only the documents containing the coronavirus terms in the abstract
# # almost all documents contain the term in the text...
# contain_coron_in = np.array(indices)[scores>0]
# data_df_red = data_df.iloc[contain_coron_in, :].copy()
# data_df_red = data_df_red.reset_index(drop=True)
# data_df_red.to_csv("./all_doc_df.csv", index=False)
