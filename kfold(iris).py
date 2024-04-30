import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Iris.csv')
data.drop(columns=['id'], inplace=True)

x = data.iloc[:, 0:4].values  # Convert DataFrame to numpy array
y = data.iloc[:, -1].values   # Convert DataFrame to numpy array

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_train, y_train)
print("LOGISTIC REGRESSION:- ", lr.score(x_test, y_test))

svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
print("SUPPORT VECTOR MACHINE:- ", svm.score(x_test, y_test))

rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
print("RANDOM FOREST:- ", rf.score(x_test, y_test))

'''
LOGISTIC REGRESSION:-  0.98
SUPPORT VECTOR MACHINE:-  0.98
RANDOM FOREST:-  0.98
'''
print('======================================================================================================')
def getscore(model,x_train, x_test, y_train, y_test ):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)


folds = StratifiedKFold(n_splits=3)

score_logistic = []
score_svm = []
score_rf = []

for train_index, test_index in folds.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    score_svm.append(getscore(SVC(gamma='auto'), x_train, x_test, y_train, y_test))
    score_logistic.append(getscore(LogisticRegression(solver='liblinear'), x_train, x_test, y_train, y_test))
    score_rf.append(getscore(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))

print("LOGISTICS SCORE:- ", score_logistic)
print("SVM SCORE:- ", score_svm)
print("RANDOM FOREST SCORE:- ", score_rf)
'''
LOGISTICS SCORE:-  [0.96, 0.96, 0.94]
SVM SCORE:-  [0.98, 0.98, 0.96]
RANDOM FOREST SCORE:-  [0.98, 0.94, 0.94]
'''
print('======================================================================================================')
#cross_val_predict

logR = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), x, y, cv=3)
print("Logistic Regression:- ",logR)

ranF = cross_val_score(RandomForestClassifier(n_estimators=40), x, y, cv=3)
print("RandomForest:- ", ranF)

SupVecMec = cross_val_score(SVC(gamma='auto'), x, y, cv=3)
print("Support Vector Mechine:- ", SupVecMec)

'''
Logistic Regression:-  [0.96 0.96 0.94]
RandomForest:-  [0.98 0.94 0.98]
Support Vector Mechine:-  [0.98 0.98 0.96]
'''

print('======================================================================================================')

score_1 = cross_val_score(RandomForestClassifier(n_estimators=100), x, y, cv=10)
avg = np.average(score_1)
print(f"RFC AVERAGE:- {avg}") #AVERAGE:- 0.9471322160148976


score_2 = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), x, y, cv=10)
avge = np.average(score_2)
print(f"LR AVERAGE:- {avge}")

score_3 = cross_val_score(SVC(gamma='auto'), x, y, cv=10)
average = np.average(score_3)
print(f"SVM AVERAGE:- {average}")

'''
RFC AVERAGE:- 0.96
LR AVERAGE:- 0.9533333333333334
SVM AVERAGE:- 0.9800000000000001
'''