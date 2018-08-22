sentiment_dict = {}
for each_line in open('dict.txt'):
	word,score = each_line.split('\t')
	senti_dict[word] = int(score)
	txt = open("NLPdataEx5data_senti_analyze.txt","r")
	for i in txt:
		word_tokens = word_tokenize(i)
word_token = word_tokens.lower().split()

print(word_token)
print(sum( sentiment_dict.get(word_token, 0) for word in word_token ))