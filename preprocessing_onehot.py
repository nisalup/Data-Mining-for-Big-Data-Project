from parser_xml import *
import pandas as pd
import numpy as np
"""[Preprocessing]

[here we get the list of cleaned words from parser.py and turn them into features using one hot encoding]
""" 

#get cleaned data
features, collected_data = data_cleaning()
print(len(features))
features = sorted(features)
print(features)
#create new dataframe and load training labels and convert to numpy
df = pd.read_csv('train.csv')
label_array = df.values
print(label_array.shape)

#create new array with features
num_rows, _ = label_array.shape
num_cols = len(features)
print('started adding values to array')
feature_array = np.full((num_rows, num_cols), False, dtype=int)
for index, label in enumerate(label_array):
	for row_index, feature in enumerate(features):
		if feature in collected_data[label[0] - 1]:
			feature_array[index, row_index] = True

#get only labels
label_array_modified = label_array[:, 2].reshape((num_rows, 1))

print(feature_array.shape)
print(label_array_modified.shape)

#merge arrays
feature_array[:,:-1] = label_array_modified
print(feature_array.shape)
np.savetxt('train_modified.txt', feature_array, delimiter=',')


