#!/usr/bin/python3
# -*- coding: utf-8 -*-
# grabthenames.py
# grabs the names from the movie review data set

import glob
import io
import os
import pdb
import sys
import pandas as pd
import numpy as np
import re
import copy
import csv

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

from sklearn.feature_extraction import DictVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
from nltk.util import ngrams
from textblob import TextBlob

def wordgrams(text,n):
    grams = []
    for ngram in ngrams(text, n):
        grams.append(' '.join(str(i) for i in ngram))
    return grams

def get_entity(text):
    features = {}
    copyfeatures = {}
    length = len(text)
    features['length'] = length
    count = re.findall('\s+', text)
    features['spaces'] = len(count)
    features['numberofwords'] = len(text.split(" "))
    features['onegram'] = len(wordgrams(text,1))
    features['twogram'] = len(wordgrams(text, 2))
    features['threegram'] = len(wordgrams(text, 3))
    features['sentiment'] = TextBlob(text).sentiment.polarity
    redact = re.findall(r'(\u2588+\s*\u2588*\s*\u2588*\s*\u2588+)', text)
    if redact:
        features['redact'] = len(redact[0])
    else:
        features['redact'] = 0
    features['nospacelength'] = len(text.replace(" ", ""))
    copyfeatures = copy.deepcopy(features)
    return copyfeatures


def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    traindata = []
    xtrain = []
    ytrain = []
    for thefile in glob.glob(glob_text):
        file_df = pd.read_csv(thefile, sep='\t', usecols=[0,1, 2, 3],header=None,engine="python", quoting=csv.QUOTE_NONE)
        file_df.columns = ['username', 'typeofdata', 'name', 'text']
        xtrain = file_df[(file_df['typeofdata'] == 'training') |(file_df['typeofdata'] == 'validation')]['text'].tolist()
        ytrain = file_df[(file_df['typeofdata'] == 'training')|(file_df['typeofdata'] == 'validation')]['name'].tolist()
        for row in xtrain:
            traindata.append(get_entity(row))
    return traindata,ytrain

def doextractredaction(glob_text):
    testdata = []
    xtest = []
    ytest = []
    for thefile in glob.glob(glob_text):
        file_df = pd.read_csv(thefile, sep='\t', usecols=[0, 1, 2, 3], header=None, engine="python",quoting=csv.QUOTE_NONE)
        file_df.columns = ['username', 'typeofdata', 'name', 'text']
        xtest = file_df[file_df['typeofdata'] == 'testing']['text'].tolist()
        ytest = file_df[file_df['typeofdata'] == 'testing']['name'].tolist()
        # print(xtest)
        for row in xtest:
            testdata.append(get_entity(row))
    return testdata,ytest
def get_redactfeatures(text):
    features = {}
    copyfeatures = {}
    length = len(text)
    features['length'] = length
    count = re.findall('\s+', text)
    features['spaces'] = len(count)
    features['numberofwords'] = len(text.split(" "))
    features['onegram'] = len(wordgrams(text, 1))
    features['twogram'] = len(wordgrams(text, 2))
    features['threegram'] = len(wordgrams(text, 3))
    features['sentiment'] = TextBlob(text).sentiment.polarity
    redact = re.findall(r'(\u2588+\s*\u2588*\s*\u2588*\s*\u2588+)', text)
    if redact:
        features['redact'] = len(redact[0])
    else:
        features['redact'] = 0
    features['nospacelength'] = len(text.replace(" ", ""))
    # words = text.split(" ")
    # left = 0
    # right = 0
    # if len(redact) == 0:
    #     left = 0
    #     right = 0
    # else:
    #     for i in range(len(words)):
    #         if words[i] == redact[0]:
    #             left = words[i-1]
    #             if i != len(words)-1:
    #                 right = words[i+1]
    #             else:
    #                 right = 0
    # features['preword'] = left
    # features['postword'] = right
    copyfeatures = copy.deepcopy(features)
    return copyfeatures


if __name__ == '__main__':
    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
    X_train,y_train = doextraction(sys.argv[-1])
    print("xtrain",X_train)
    X_test, y_test = doextractredaction(sys.argv[-1])
    print("ytestvalue",y_test[0])
    print("xtest",X_test[0])
    vectorizer = DictVectorizer(sparse=False)
    X_train_vec = vectorizer.fit_transform(X_train)
    # Y_train_vec = LabelEncoder().fit_transform(y_train)
    X_test_vec = vectorizer.fit_transform(X_test)
    # y_test_vec = LabelEncoder().fit_transform(y_test)

    # log = LogisticRegression(max_iter=1000)
    # log.fit(X_train_vec,y_train)
    vec = RandomForestClassifier(n_estimators=100)
    vec.fit(X_train_vec, y_train)
    print("training done")
    # vec = RandomForestClassifier()
    # vec.fit(X_train_vec,y_train)
    y_pred1 = vec.predict(X_test_vec)
    print(y_pred1)
    # cuisine1 = LabelEncoder().fit(y_train).inverse_transform(y_pred1).tolist()
    # print(cuisine1)
    #
    score1 = f1_score(y_test, y_pred1, average='macro')
    print('F-1 score1 : {}'.format(np.round(score1, 4)))
    accuracy1 = accuracy_score(y_test, y_pred1)
    print("accuracy1:", accuracy1)
    precision1 = precision_score(y_test, y_pred1, average='macro')
    print("precison1:", precision1)


    # # vec = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter =5000,alpha=0.0001,solver='sgd',verbose=10, random_state=21, tol=0.0000001)
    # # vec.fit(X_train_vec, Y_train_vec)
    # knn =  KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    # knn.fit(X_train_vec, y_train)
    #
    # # y_pred2 = knn.kneighbors(X_test_vec,n_neighbors=1,return_distance=False)
    # y_pred2 = knn.predict(X_test_vec)
    # print(y_pred2)
    # cuisine2 = LabelEncoder().fit(y_train).inverse_transform(y_pred2).tolist()
    # print(cuisine2)
    #
    # score2 = f1_score(y_test_vec, y_pred2, average='micro')
    # print('F-1 score2 : {}'.format(np.round(score2, 4)))
    # accuracy2 = accuracy_score(y_test_vec, y_pred2)
    # print("accuracy2:", accuracy2)
    # precision2 = precision_score(y_test_vec, y_pred2, average='macro')
    # print("precison2:", precision2)
    #
    # clf = MultinomialNB()
    # clf.fit(X_train_vec,y_train)


