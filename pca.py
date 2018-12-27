from preprocessing_onehot import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  
"""[SVM]

[here we try to perform SVM classifier on the data]
""" 

X = np.loadtxt("train_modified.txt", delimiter=",")
y = np.loadtxt("train_modified_labels.txt", delimiter=",")[:,2]
#split train and test to verify accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#we use SVC with linear kernel
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  








