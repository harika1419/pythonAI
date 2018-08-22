import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

df = pd.read_csv("NLP.csv")
words = df.iloc[:,0]
txt = open("out_data.csv","w")
for i in words:
	wordtok = word_tokenize(i)
	wordtoks = str(wordtok)
	txt.write(wordtoks+'\n')