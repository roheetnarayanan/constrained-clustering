import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from nltk.corpus import stopwords
ENG_STOP = stopwords.words("ENGLISH")
from sklearn.decomposition import TruncatedSVD
from copkmeans.cop_kmeans import cop_kmeans
from nltk.stem import *
import gensim
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import spacy


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc =spacy.load("en_core_web_sm")(" ".join(sent)) 
        texts_out.append([str(token.lemma_) for token in doc if token.pos_ in allowed_postags])
    return texts_out

def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>') 
    cleantext = re.sub(CLEANR, '', raw_html)
    cleantext = cleantext.replace("/"," ").replace("(","").replace(")","")
    res = []
    l_res = []
    for word in cleantext.split(" "):
        word = word.lower()
        if word.isalpha() and word not in ENG_STOP:
            res.append(word)
    return res

def get_bow(text,max_features):
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False,max_features=max_features)
    bow = vectorizer.fit_transform(text)
    text_d = bow.toarray()
    temp = pd.DataFrame(text_d, columns = vectorizer.get_feature_names())
    return temp

def get_tfidf(text,max_features):
    vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False,max_features=max_features)
    bow = vectorizer.fit_transform(text)
    text_d = bow.toarray()
    temp = pd.DataFrame(text_d, columns = vectorizer.get_feature_names())
    return temp
