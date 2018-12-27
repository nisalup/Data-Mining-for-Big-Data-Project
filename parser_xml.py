import xml.etree.ElementTree
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string
"""[Parsing]

[here we parse the data from xml and clean them up and remove unwanted words. The process is commented step by step]
"""

def data_cleaning():

	tokenized_collection = []
	#loop through all xml files
	for file_name in range(5000):
		e = xml.etree.ElementTree.parse('./data/' + str(file_name + 1) + '.xml').getroot()
		
		#check if empty
		if not e.text:
			tokenized_collection.append([])
		else:
			xml_text = word_tokenize(e.text)
			xml_text_stop = []

			#remove empty words, and remove punctuations
			xml_text_stop = list(filter(None, xml_text))
			for index, w in enumerate(xml_text_stop):
				xml_text_stop[index] = re.sub(r'[^\w\s]','',w)

			#stop words processing
			xml_text_stop_fin = []
			stopWords = set(stopwords.words('english'))
			for w in xml_text_stop:
			   	if w not in stopWords:
			   		xml_text_stop_fin.append(w)

			#stem words processing
			ps = PorterStemmer()
			xml_text_stop_fin = [ps.stem(a) for a in xml_text_stop_fin]

			#remove words with length lesser than 3
			xml_text_stop_fin = [word for word in xml_text_stop_fin if len(word) >= 3]

			#remove duplicates
			xml_text_stop_cleaned = list(set(xml_text_stop_fin))

			#remove numbers and add hasNum feature
			for index, w in enumerate(xml_text_stop_cleaned):
				if str(w).isdigit():
					del xml_text_stop_cleaned[index]
					if 'hasNum' not in xml_text_stop_cleaned:
						xml_text_stop_cleaned.append('hasNum')


			#print(xml_text_stop_fin)
			tokenized_collection.append(xml_text_stop_cleaned)

	#remove features that are mentioned less than 5 times in the dataset
	#create feature set
	feature_set = set()
	for item in tokenized_collection:
		for w in item:
			if w not in feature_set:
				feature_set.add(w)
	#lookup frequencies of occurences
	for feature in feature_set.copy():
		feature_count = 0
		for tokenized_collection_item in tokenized_collection:
			feature_count += tokenized_collection_item.count(feature)
		if feature_count < 5:
			for tokenized_collection_item_second in tokenized_collection:
				while feature in tokenized_collection_item_second: tokenized_collection_item_second.remove(feature)
			feature_set.remove(feature)


	print(len(feature_set))

	#print(tokenized_collection)
	return feature_set, tokenized_collection

#data_cleaning()
