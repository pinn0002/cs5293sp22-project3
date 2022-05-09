Name: Divya Sai Pinnamaneni

Run Procedure:

Install all the below specified libraries using the following command pipenv install modulename.

Install pipenv using pip install pipenv

Run the project using pipenv run python unredactor.py unredacted.tsv where unredactor.py is my python script and unredacted.tsv is name of input file

Run testfiles using pipenv run python -m pytest

Expected bugs

Any other file format except .tsv will not be accepted as input file format.
With the concept function, no other sentences can be redacted except the words that are acquired using lemma_names, wordnet.synsets functions.

While calculating precision score one zerodivision warning is created
Libraries used in this project:

glob, sys, nltk, sklearn, pandas, copy, re, numpy, timeblob, 


Assumptions:

For performing this project, below are my assumptions
1. As I already have training, testing, validation data separately so to perform unredaction I believed training data to be used for training machine with the corresponding names.
2. Testing sentences are used for prediction of redacted word 
3. So, I have assumed last column from unredacted.tsv need to be trained first with the corresponding names from the previous column.
4. All training sentences in unredacted.tsv are used for training and testing sentences are used for calculating predictions.
5. Output of redacted names to be displayed along with precision accuracy and f-1scores need to be displayed directly to the terminal.
