import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk

df = pd.read_csv("NLP.csv")
sente = df.iloc[:,0]
txt = open("out_data.csv","w")
for i in sente:
	sentence = sent_tokenize(i)
	sentes = str(sentence)
	txt.write(sentes+'\n')
# ne_chunk for the sentence.
	nechunk_sentence = ne_chunk(sentence)
	print(sentes)
	nechunk_sentence.draw()