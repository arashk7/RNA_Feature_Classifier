import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

features1 = pd.read_csv('../Dataset/Sample1.csv')
features1.head()
features2 = pd.read_csv('../Dataset/Sample2.csv')
features2.head()
features = pd.concat([features1, features2])
features.head()
# print(features)

features1 = features1.replace('mod', 0)
features1 = features1.replace('unm', 1)
features1 = features1.replace(np.nan, 0, regex=True)

# print(features)
X_train = features1[['q1', 'q2', 'q3', 'q4', 'q5', 'mis1', 'mis2', 'mis3', 'mis4', 'mis5']].astype(float)
y_train = features1['sample'].astype(int)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)

features2 = features2.replace('mod', 0)
features2 = features2.replace('unm', 1)
features2 = features2.replace(np.nan, 0, regex=True)

# print(features)
X_test = features2[['q1', 'q2', 'q3', 'q4', 'q5', 'mis1', 'mis2', 'mis3', 'mis4', 'mis5']].astype(float)
y_test = features2['sample'].astype(int)


sc = MinMaxScaler(feature_range=(0,1))
X_test = sc.fit_transform(X_test)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)

#Classifier
from sklearn.ensemble import BaggingClassifier
clf = ExtraTreeClassifier()
# clf = BaggingClassifier(clf, random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()
