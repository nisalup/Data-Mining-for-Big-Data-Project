from parser_xml import *
import pandas as pd
import numpy as np
"""[Preprocessing]

[here we get the list of cleaned words from parser.py and turn them into features using one hot encoding]
""" 
def data_preprocessing():
	#get cleaned data
	features, collected_data = data_cleaning()
	print(len(features))
	features = sorted(features)
	print(features)
	#create new dataframe and load training labels and convert to numpy
	df = pd.read_csv('train.csv')
	label_array = df.values
	print(label_array.shape)

	#same way, load test data file
	df1 = pd.read_csv('test.csv')
	pred_array = df1.values

	#create new array with features for training
	num_rows, _ = label_array.shape
	num_cols = len(features)
	print('started adding values to training array')
	feature_array = np.zeros((num_rows, num_cols), dtype=int)

	print('nisal ' + str(label_array[0,2]))

	for index, label in enumerate(label_array):
		for row_index, feature in enumerate(features):
			if feature in collected_data[label[0] - 1]:
				feature_array[index, row_index] = 1


	print(feature_array.shape)
	print(label_array.shape)
	np.savetxt('train_modified.txt', feature_array, delimiter=',')
	np.savetxt('train_modified_labels.txt', label_array[:,2], delimiter=',', fmt='%s')

	#create new array with features for testing
	num_rows_test, _ = pred_array.shape
	print('started adding values to test array')
	feature_array_test = np.zeros((num_rows_test, num_cols), dtype=int)

	for index, label in enumerate(pred_array):
		for row_index, feature in enumerate(features):
			if feature in collected_data[label[0] - 1]:
				feature_array_test[index, row_index] = 1

	np.savetxt('test_modified.txt', feature_array_test, delimiter=',')


data_preprocessing()
