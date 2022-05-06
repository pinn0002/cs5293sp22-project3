#!/usr/bin/python3
# -*- coding: utf-8 -*-
# grabthenames.py
# grabs the names from the movie review data set

import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



def get_entity(text):
    """Prints the entity inside of the text."""
    names = []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                # print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                names.append(' '.join(c[0] for c in chunk.leaves()))
        for person in names:
            # hide names in textfiles
            sent = sent.replace(person, '\u2588' * len(person))

def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    alltext = []
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            alltext.append(text)
    # print(alltext)
    vec = TfidfVectorizer()
    fitvec = vec.fit_transform(alltext)
    # print("fittedarray",fitvec)
    idf = vec.idf_
    print("idf",idf)
    names = vec.get_feature_names_out()
    # print("names",names)

    get_entity(text)


if __name__ == '__main__':
    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
    doextraction(sys.argv[-1])