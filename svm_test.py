import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
"""[Main code]

[here we predict the labes for the test data given]
""" 
#first import training data
X_train = np.loadtxt("train_modified.txt", delimiter=",")
y_train = np.loadtxt("train_modified_labels.txt", delimiter=",")

#get testing data
X_test = np.loadtxt("test_modified.txt", delimiter=",")

#we use SVC with linear kernel as before
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(y_pred.shape)
#loop through given test data file and fill it
df1 = pd.read_csv('test.csv')
prediction_file = df1.values
print(prediction_file.shape)

for index, prediction_line in enumerate(prediction_file):
	prediction_line[2] = int(y_pred[index])

#save it
df = pd.DataFrame(prediction_file, columns=['id','file','earnings'])
df.to_csv('test_final.csv')