from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

"""[Main code]

[here we predict the labes for the test data given]
"""
#first import training data
X_train = np.loadtxt("train_modified.txt", delimiter=",")
y_train = np.loadtxt("train_modified_labels.txt", delimiter=",")

#get testing data
X_test = np.loadtxt("test_modified.txt", delimiter=",")

#we use sklearn.naive_bayes.MultinomialNB estimator:
model_multinomialDB = MultinomialNB()
model_multinomialDB.fit(X_train, y_train)

y_pred_multinomialNB = model_multinomialDB.predict(X_test)

print(y_pred_multinomialNB.shape)
#loop through given test data file and fill it
df1 = pd.read_csv('test.csv')
prediction_file = df1.values
print(prediction_file.shape)

for index, prediction_line in enumerate(prediction_file):
	prediction_line[2] = int(y_pred_multinomialNB[index])

#save it
df = pd.DataFrame(prediction_file, columns=['id','file','earnings'])
df.to_csv('test_final_multinomialNB_classifier.csv')