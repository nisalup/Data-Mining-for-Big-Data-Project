from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
"""[naive_bayes]

[here we try to perform naive_bayes.GaussianNB estimator on the training data]
"""
X = np.loadtxt("train_modified.txt", delimiter=",")
y = np.loadtxt("train_modified_labels.txt", delimiter=",")

#using PCA to reduce dimensionality to lower dimension
#but first let us try to find the perfect n_compoents to use
scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(X)
pca = PCA().fit(data_rescaled)

plt.figure(1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Variance Change with Number of Components')
plt.show()

#since it shows from the graph that we can reduce the dimensionality, we will try this with PCA
# n_components=10 -> precision=0.9391, n_components=13 -> precision=0.9425, n_components=14 -> precision=0.94333
# n_components=15 -> precision=0.939, n_components=16 -> precision=0.875
pca_scaled = PCA(n_components=14)
pca_scaled.fit(X)
X = pca_scaled.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

## Normalized Data from 0 to 1, cause Input X must be non-negative for MultinomialNB estimator
# ValueError: Input X must be non-negative for MultinomialNB estimator
# we try to use class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
# in this case, we have test set accuracy: 0.7883333333333333
# so we decided to use GaussianNB
#print("X train:\n", X_train)
#print("X train:\n", X_train)
#min_max_scaler = MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.transform(X_test)
#print("X train after scale:\n", X_train)
#print("X test after scale:\n", X_train)

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


fig = plt.figure(2)
ax = fig.add_subplot(121)
cax = ax.matshow(cm_gaussianNB)
plt.title('Confusion matrix n=2')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()