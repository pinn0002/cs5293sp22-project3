Name: Divya Sai Pinnamaneni

Run Procedure:

Install all the below specified libraries using the following command pipenv install modulename.

Install pipenv using pip install pipenv

Run the project using pipenv run python unredactor.py unredacted.tsv where unredactor.py is my python script and unredacted.tsv is name of input file

Run testfiles using pipenv run python -m pytest

Expected bugs

Any other file format except .tsv will not be accepted as input file format.

While calculating precision score one zerodivision warning is created.


Libraries used in this project:

glob, sys, nltk, sklearn, pandas, copy, re, numpy, timeblob, csv, textblob


Assumptions:

For performing this project, below are my assumptions
1. As I already have training, testing, validation data separately so to perform unredaction I believed training data to be used for training machine with the corresponding names.
2. Testing sentences are used for prediction of redacted word 
3. So, I have assumed last column from unredacted.tsv need to be trained first with the corresponding names from the previous column.
4. All training sentences in unredacted.tsv are used for training and testing sentences are used for calculating predictions.
5. Output of redacted names to be displayed along with precision accuracy and f-1scores need to be displayed directly to the terminal.


Project Objective:
Main aim of this project is to predict the name which is redacted using blocks in the sentences. To make machine understand the redacted words, I'm required to train machine with the sentences containing redacted words and corresponding output name words for redacted word. After training the model, provide sentence and predict expected redacted word.

unredacted.tsv is the input file which is separated with tab space between columns.
Output of the redacted sentences is displayed to the terminal in the form of list.

Dataset used: unredacted.tsv is used which is collection of collobarative sentences from every member of this course.

Functions:

doextraction(sys.argv[-1]):

This function takes the file input from sys.argv[-1] which is nothing but the file used in run statement. For example,in  pipenv run python unredactor.py unredacted.tsv. unredacted.tsv is the position for sys.argv[-1]. Expected output from this function should return related xtrain and ytrain variables for fitting the model into.

This function reads unredacted.tsv file using pd.read_csv as a dataframe and considers only columns with names and sentences with redacted text but only considers sentences if they are training related.
names column is created as list of names for sending it as ytrain data. sentences with redacted text are sent to get_entity function for retrieving features of each sentence.And result from all features are appended to a lsit for considering it to be xtrain data.

get_entity(text):

This function takes sentence with redacted text as input and extract features of this sentence. I have used following features for detecting a sentence 
Features used:
Length of sentence
Count of spaces in a sentence

Number of words in a sentence

One gram, Two gram, Three grams length of sentences are used. 

For obtaing n-grams I have used a function wordgrams(text,n) which takes sentences and number of grams required as arguments and provide list of grams created as output and I have obtained this functionality using ngrams package available from nltk.util

Sentiment score of a sentence - For getting sentiment score I have used TextBlob function from textblob package  and TextBlob(text).sentiment.polarity is used for obtaining sentiment score of a sentence.

Redacted words - Use a regex pattern to find redacted words in a sentence and count number of those redacted words. If there are no redacted words by default consider as '0'.

length of a  sentence with no space - Find length of sentence by removing spaces in between

All these features are added to a dictionary and copied to another dictionary for not missing any of the features of sentences.

This function would return dictionary of all above features

doextractredaction(sys.argv[-1]):

This function would also take unredacted.tsv as input and provide Xtest and Ytest as output. Similar to the doextraction it reads unredacted.tsv file using pd.read_csv as a dataframe and considers only columns with names and sentences with redacted text but only consider sentences if they are testing related.
names column is created as list of names for sending it as ytest data. sentences with redacted text are sent to get_redactfeature function for retrieving features of each sentence. And result from all features are appended to a lsit for considering it to be xtest data. 

ytest values are actual result of predicted value.
xtest values are features of testing sentences.

get_redactfeatures(text):

This function takes sentence with redacted text as input and extract features of this sentence. I have used following features for detecting a sentence 
Features used:
Length of sentence
Count of spaces in a sentence

Number of words in a sentence

One gram, Two gram, Three grams length of sentences are used. 

For obtaing n-grams I have used a function wordgrams(text,n) which takes sentences and number of grams required as arguments and provide list of grams created as output and I have obtained this functionality using ngrams package available from nltk.util

Sentiment score of a sentence - For getting sentiment score I have used TextBlob function from textblob package  and TextBlob(text).sentiment.polarity is used for obtaining sentiment score of a sentence.

Redacted words - Use a regex pattern to find redacted words in a sentence and count number of those redacted words. If there are no redacted words by default consider as '0'.

length of a  sentence with no space - Find length of sentence by removing spaces in between

All these features are added to a dictionary and copied to another dictionary for not missing any of the features of sentences.

This function would return dictionary of all above features.

Use DictVectorizer to fit_transform of xtrain and xtest data for informing machine about the features in numerical formats. 

Use RandomforestClassifier with n_estimators = 100 to fit the model with xtrain and ytrain data and obtain prediction for xtest.

For obtaining, F-1 score precision and recall use functions from sklearn.metrics and pass arguments as actual values and predicted values.

Though different classifiers like LogisticRegression, KNeighboursclassifier but used RandomforestClassifier as they give better results.

Testcases:

def test_doextraction():
    checks original doextraction function returns list or not.
def test_entity():
    checks original get_entity function and returns dictionary or not.
def test_extraction():
    checks original get_redactfeatures and returns dictionary or not.    
def test_entityfeatures():
    checks original doextractredaction and check if return is empty or not.




