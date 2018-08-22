from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import stem
from nltk.stem import WordNetLemmatizer

txt = open("NLP3&4.txt","r")
for i in txt:
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(i)
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	filtered_sentence = []
	for w in word_tokens:
		if w not in stop_words:
			filtered_sentence.append(w)
	print("Word tokenize of sentence")
	print(word_tokens)
	print("Excluding stops words : ")
	print(filtered_sentence)
#Performed Snowball Stemmer
	ss = SnowballStemmer('english')
	print("Snowball STemmer is applied :")
	for word in word_tokens:
		print(ss.stem(word))
#Performed Porter Stemmer
	ps = PorterStemmer()
	print("Porter Stemer is applied :")
	for word in word_tokens:
		print(ps.stem(word))
	Lemmatizer = WordNetLemmatizer()
	for word in word_tokens:
		pritn("Lemmatizing of words :")
		print(Lemmatizer.lemmatize(word))
