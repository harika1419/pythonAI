from nltk.corpus import stopwords
sentiment_dict = {}
for each_line in open('dict.txt'):
	word,score = each_line.split('\t')
	sentiment_dict[word] = int(score)
	wordst=[]
	txt = open("NLPdataEx5data_senti_analyze.txt","r")
	for i in txt:
		stop_words = set(stopwords.words('english'))
		i=i.strip()
		words=i.split()
		for i in words:
			wordst.append(i)
			filtered_sentence = [w for w in wordst if not w in stop_words]
			filtered_sentence = []
			for w in wordst:
				if w not in stop_words:
					filtered_sentence.append(w)
print(wordst)
print(filtered_sentence)
print(sum( sentiment_dict.get(word, 0) for word in filtered_sentence ))