from parser_xml import *
import pandas as pd
"""[Preprocessing]

[here we get the list of cleaned words from parser.py and turn them into features using one hot encoding]
""" 

#get cleaned data
collected_data = data_cleaning()

#create new dataframe and load training labels
df = pd.read_csv('train.csv')

#loop through cleaned data and add to df with one hot
for index, row in df.iterrows():
	print(int(row['id']) - 1)
	if collected_data[int(row['id']) - 1]:
		for word in collected_data[int(row['id']) - 1]:
			df[word][index] = 1
print(df)

