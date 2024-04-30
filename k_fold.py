import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_train, y_train)
# print("LOGISTIC REGRESSION:- ",lr.score(x_test,y_test))
'''
solver='liblinear': This parameter specifies the algorithm to be used in the optimization problem. 
'liblinear' is suitable for small datasets and supports both L1 and L2 regularization. 
It's particularly efficient for binary classification tasks.
multi_class='ovr': This parameter specifies the strategy to use when the target variable has multiple classes. 
'ovr' stands for "one-vs-rest" and it means that scikit-learn will fit one binary classifier per class, with all the samples from that class considered as positive and the rest as negatives. 
This strategy is also known as one-vs-all.
'''

svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
# print("SUPPORT VECTOR MACHINE:- ",svm.score(x_test, y_test))

rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
# print("RANDOM FOREST:- ",rf.score(x_test, y_test))

'''
LOGISTIC REGRESSION:-  0.9595959595959596
SUPPORT VECTOR MACHINE:-  0.25925925925925924
RANDOM FOREST:-  0.968013468013468
'''

kf = KFold(n_splits=3)
for train_index, test_index, in kf.split([1,2,3,4,5,6,7,8,9]):
    # print(train_index, test_index)
    '''
    [3 4 5 6 7 8] [0 1 2]
    [0 1 2 6 7 8] [3 4 5]
    [0 1 2 3 4 5] [6 7 8]
    '''



def getscore(model,x_train, x_test, y_train, y_test ):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)

# print(getscore(SVC(),x_train, x_test, y_train, y_test)) #0.9966329966329966

folds = StratifiedKFold(n_splits=3)

score_logistic = []
score_svm = []
score_rf = []

for train_index,test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    score_svm.append(getscore(SVC(gamma='auto'),X_train, X_test, y_train, y_test))
    score_logistic.append(getscore(LogisticRegression(solver='liblinear'),X_train, X_test, y_train, y_test))
    score_rf.append(getscore(RandomForestClassifier(n_estimators=40),X_train, X_test, y_train, y_test))

# print("LOGISTICS SCORE:- ",score_logistic)
# print("SVM SCORE:- ",score_svm)
# print("RANDOM FOREST SCORE:- ",score_rf)

'''
StratifiedKFold(n_splits=3)

LOGISTICS SCORE:-  [0.8948247078464107, 0.9532554257095158, 0.9098497495826378]
SVM SCORE:-  [0.3806343906510851, 0.41068447412353926, 0.5125208681135225]
RANDOM FOREST SCORE:-  [0.9449081803005008, 0.9565943238731218, 0.9282136894824707]
'''

#cross_val_predict

logR = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target, cv=3)
print("Logistic Regression:- ",logR)

ranF = cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3)
print("RandomForest:- ", ranF)

SupVecMec = cross_val_score(SVC(gamma='scale'), digits.data, digits.target, cv=3)
print("Support Vector Mechine:- ", SupVecMec)


'''
Logistic Regression:-  [0.89482471 0.95325543 0.90984975]
RandomForest:-  [0.93823038 0.94323873 0.92320534]
Support Vector Mechine:-  [0.38063439 0.41068447 0.51252087]
'''

print('----------------------------------------------------------------------------------------------')

score_1 = cross_val_score(RandomForestClassifier(n_estimators=100), digits.data, digits.target, cv=10)
avg = np.average(score_1)
print(f"RFC AVERAGE:- {avg}") #AVERAGE:- 0.9471322160148976


score_2 = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target, cv=10)
avge = np.average(score_2)
print(f"LR AVERAGE:- {avge}")

score_3 = cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=10)
average = np.average(score_3)
print(f"SVM AVERAGE:- {average}")