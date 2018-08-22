from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

txt = open("NLP3&4.txt","r")

for i in txt:
	poswords = pos_tag(word_tokenize(i))
	print(poswords)