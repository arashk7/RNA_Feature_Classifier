from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd



n_samples = 10000
sam_list = list()
acc_list = list()
X, Y = make_classification(n_features=1, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, n_samples=10000)
for i in range(0, 50):
    n_samples += 1000
    sam_list.append(n_samples)
    X1, Y1 = make_classification(n_features=1, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, n_samples=1000)
    X=[*X,*X1]
    Y=[*Y,*Y1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)
    clf = RandomForestClassifier(n_estimators=80)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    acc_list.append(acc)

fig = plt.figure()
ax = plt.axes()

# x = np.linspace(0, 10, 1000)
ax.plot(sam_list, acc_list)
plt.show()

