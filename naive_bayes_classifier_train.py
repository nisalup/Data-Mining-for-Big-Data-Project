from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

"""[naive_bayes]

[here we try to perform naive_bayes.GaussianNB estimator on the training data]
"""

X = np.loadtxt("train_modified.txt", delimiter=",")
y = np.loadtxt("train_modified_labels.txt", delimiter=",")
print(y[0])

#split train and test to verify accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#we use sklearn.naive_bayes.GaussianNB estimator:
model_gaussianNB = GaussianNB()
model_gaussianNB.fit(X_train, y_train)

y_pred_gaussianNB = model_gaussianNB.predict(X_test)
print("Naive Bayes GaussianNB Classifier:\n")
print("test set accuracy:", np.mean(y_pred_gaussianNB == y_test))

#confusion matrix
cm_gaussianNB = confusion_matrix(y_test,y_pred_gaussianNB)
print("confusion matrix:\n", cm_gaussianNB)
print("classification report:\n", classification_report(y_test,y_pred_gaussianNB))

#we use sklearn.naive_bayes.MultinomialNB estimator:
model_multinomialDB = MultinomialNB()
model_multinomialDB.fit(X_train, y_train)

y_pred_multinomialNB = model_multinomialDB.predict(X_test)
print("Naive Bayes MultinomialNB Classifier:\n")
print("test set accuracy:", np.mean(y_pred_multinomialNB == y_test))

#confusion matrix
cm_multinomialNB = confusion_matrix(y_test,y_pred_multinomialNB)
print("confusion matrix:\n", cm_multinomialNB)
print("classification report:\n", classification_report(y_test,y_pred_multinomialNB))