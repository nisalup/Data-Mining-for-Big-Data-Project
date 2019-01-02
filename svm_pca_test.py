import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
"""[SVM]

[here we try to perform SVM classifier with PCA reduction on the training data]
""" 

X = np.loadtxt("train_modified.txt", delimiter=",")
y = np.loadtxt("train_modified_labels.txt", delimiter=",")

#using PCA to reduce dimensionality to lower dimension
#but first let us try to find the perfect n_compoents to use
scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(X)
pca = PCA().fit(data_rescaled)

plt.figure(0)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Variance Change with Number of Components')
plt.show()

#since it shows from the graph that we can reduce the dimensionality, we will try this with PCA
# n_components=2 -> precision=0.85, n_components=14 -> precision=0.94
pca_scaled = PCA(n_components=14)
pca_scaled.fit(X)
X = pca_scaled.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

svclassifier_pca = SVC(kernel='linear')
svclassifier_pca.fit(X_train, y_train)

y_pred = svclassifier_pca.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)  
print(classification_report(y_test,y_pred)) 

fig = plt.figure()
ax = fig.add_subplot(121)
cax = ax.matshow(cm)
plt.title('Confusion matrix n=2')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()





