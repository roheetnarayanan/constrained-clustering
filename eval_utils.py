import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import CoherenceModel
import pandas as pd
import numpy as np
import warnings
from nltk.stem import *
from nltk.corpus import stopwords
ENG_STOP = stopwords.words("ENGLISH")
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from sklearn.decomposition import TruncatedSVD
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from scipy.spatial.distance import euclidean
from copkmeans.cop_kmeans import cop_kmeans
from sklearn.metrics import silhouette_score

def get_constraints(similarity_df, N):
    n_ml = len(similarity_df) + N*2
    n_cl = N*2
    Top_vals = similarity_df.unstack().sort_values(ascending=False)[:n_ml]
    Bottom_vals = similarity_df.unstack().sort_values(ascending=True)[:n_cl]
    a =list(Top_vals.index.sort_values())
    b = list(Bottom_vals.index.sort_values())
    c = [(i,j) for (i,j) in a if i!=j]
    cl = list(set([tuple(sorted(sub)) for sub in b]))
    ml =list(set([tuple(sorted(sub)) for sub in c]))
    return ml, cl

def get_silhouette_list(dt,cluster_dict):
    cluster_list = list(cluster_dict.values())
    num_k = list(cluster_dict.keys())
    sil_list = []
    for cl in cluster_list:
        s = silhouette_score(dt,cl)
        sil_list.append(s)
    return sil_list    
