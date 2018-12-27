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
			print(file_name)
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
			print(xml_text_stop_fin)


	return tokenized_collection

#data_cleaning()
