Name: Divya Sai Pinnamaneni

Run Procedure:

Install all the below specified libraries using the following command pipenv install modulename.

Install pipenv using pip install pipenv

Run the project using pipenv run python unredactor.py unredacted.tsv where unredactor.py is my python script and unredacted.tsv is name of input file

Run testfiles using pipenv run python -m pytest

Expected bugs

Any other file format except .tsv will not be accepted as input file format.
With the concept function, no other sentences can be redacted except the words that are acquired using lemma_names, wordnet.synsets functions.

Libraries used in this project:

glob, sys, , nltk, numpy, pathlib 