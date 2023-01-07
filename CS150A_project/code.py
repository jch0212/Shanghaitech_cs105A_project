import pandas as pd
import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import csv
def DataPreProcess(data):
    # first convert the label and then split the data
    labelEncoder = sklearn.preprocessing.LabelEncoder()
    x1 = labelEncoder.fit_transform(data['Student'])
    x1 = pd.Series(x1.tolist())
    x2 = labelEncoder.fit_transform(data['Problem'])
    x2 = pd.Series(x2.tolist())
    x3 = labelEncoder.fit_transform(data['Step'])
    x3 = pd.Series(x3.tolist())
    x5 = data['Student_FACR']
    x5 = pd.Series(x5.tolist())
    x6 = data['Step_FACR']
    x6 = pd.Series(x6.tolist())
    x7 = data['Problem_FACR']
    x7 = pd.Series(x7.tolist())
    X = pd.concat([x1,x2,x3,x5,x6,x7], axis=1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, data['**Correct First Attempt**'], test_size=0.4, random_state=1)
    return Xtrain, Xtest, ytrain, ytest


# Here we uses 5 Classifier:
def Ada(Xtrain, Xtest, ytrain, ytest):
    clf = AdaBoostClassifier()
    clf.fit(Xtrain, ytrain)
    pre = clf.predict(Xtest)
    return sklearn.metrics.accuracy_score(ytest, pre), sklearn.metrics.classification_report(ytest, pre), pre

def GBDT(Xtrain, Xtest, ytrain, ytest):
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(Xtrain, ytrain)
    pre = clf.predict(Xtest)
    return sklearn.metrics.accuracy_score(ytest, pre), sklearn.metrics.classification_report(ytest, pre), pre


def RFC(Xtrain, Xtest, ytrain, ytest):
    rf = RandomForestClassifier(max_depth=7,max_features=0.44564993926538743,min_samples_split=int(6.972807143834928),n_estimators=int(55.73671041246315))
    rf.fit(Xtrain, ytrain)
    pre = rf.predict(Xtest)
    return sklearn.metrics.accuracy_score(ytest, pre),sklearn.metrics.classification_report(ytest, pre), pre


def KNN(Xtrain, Xtest, ytrain, ytest):
    qd = sklearn.neighbors.KNeighborsClassifier()
    qd.fit(Xtrain, ytrain)
    pre = qd.predict(Xtest)
    return sklearn.metrics.accuracy_score(ytest, pre), sklearn.metrics.classification_report(ytest, pre), pre


def naiveBayes(Xtrain, Xtest, ytrain, ytest, Xfinal):
    nB = LGBMClassifier()
    nB.fit(Xtrain, ytrain)
    pre = nB.predict(Xtest)
    result = nB.predict(Xfinal)
    return sklearn.metrics.accuracy_score(ytest, pre), sklearn.metrics.classification_report(ytest, pre), pre, result


def NN(Xtrain, Xtest, ytrain, ytest):
    ld = MLPClassifier()
    ld.fit(Xtrain, ytrain)
    pre = ld.predict(Xtest)
    return sklearn.metrics.accuracy_score(ytest, pre), sklearn.metrics.classification_report(ytest, pre)


def Vote5(Xtrain, Xtest, ytrain, ytest, Xfinal):
    pre = np.zeros(len(Xtest))
    fin = np.zeros(len(Xfinal))
    Predicts = []
    final = []
    nB = LGBMClassifier()
    nB.fit(Xtrain, ytrain)
    Predicts.append(nB.predict(Xtest))
    final.append(nB.predict(Xfinal))
    qd = sklearn.neighbors.KNeighborsClassifier()
    qd.fit(Xtrain, ytrain)
    Predicts.append(qd.predict(Xtest))
    final.append(qd.predict(Xfinal))
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(Xtrain, ytrain)
    Predicts.append(rf.predict(Xtest))
    final.append(rf.predict(Xfinal))
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(Xtrain, ytrain)
    Predicts.append(clf.predict(Xtest))
    final.append(clf.predict(Xfinal))
    clf1 = AdaBoostClassifier()
    clf1.fit(Xtrain, ytrain)
    Predicts.append(clf1.predict(Xtest))
    final.append(clf1.predict(Xfinal))
    Weights = [1.2, 0.8, 0.7, 1.15, 1.15]
    for i in range(len(Xtest)):
        count = 0
        for j in range(5):
            if Predicts[j][i] == 1:
                count += Weights[j]
        if count >= 2.3:
            pre[i] = 1

    for i in range(len(Xfinal)):
        count = 0
        for j in range(5):
            if final[j][i] == 1:
                count += Weights[j]
        if count >= 2.3:
            fin[i] = 1

    return sklearn.metrics.accuracy_score(ytest, pre), sklearn.metrics.classification_report(ytest, pre), pre, fin




books = pd.read_csv('input_data.csv')
x = pd.read_csv('test_data.csv')
jch = sklearn.preprocessing.LabelEncoder()
x1 = jch.fit_transform(x['Student'])
x1 = pd.Series(x1.tolist())
x2 = jch.fit_transform(x['Problem'])
x2 = pd.Series(x2.tolist())
x3 = jch.fit_transform(x['Step'])
x3 = pd.Series(x3.tolist())
x5 = x['Student_FACR']
x5 = pd.Series(x5.tolist())
x6 = x['Step_FACR']
x6 = pd.Series(x6.tolist())
x7 = x['Problem_FACR']
x7 = pd.Series(x7.tolist())
Xfinal = pd.concat([x1,x2,x3,x5,x6,x7], axis=1)



Xtrain, Xtest, ytrain, ytest = DataPreProcess(books)

with open("Result2.txt", "w") as f:
    write = ''
    ac1, rep1, pre1 = Ada(Xtrain, Xtest, ytrain, ytest)
    ac2, rep2, pre2 = RFC(Xtrain, Xtest, ytrain, ytest)
    ac3, rep3, pre3 = KNN(Xtrain, Xtest, ytrain, ytest)
    ac4, rep4, pre4, bresult = naiveBayes(Xtrain, Xtest, ytrain, ytest, Xfinal)
    ac5, rep5 = NN(Xtrain, Xtest, ytrain, ytest)
    ac6, rep6, pre5 = GBDT(Xtrain, Xtest, ytrain, ytest)
    # apply vote algorithm
    ac7, rep7, pre7, result = Vote5(Xtrain, Xtest, ytrain, ytest, Xfinal)
    write += 'AdaBoost Classifier Accuracy:' + str(ac1) + '\n'
    write += 'Random Forest Classifier Accuracy:' + str(ac2) + '\n'
    write += 'K Nearest Neighbor Classifier Accuracy:' + str(ac3) + '\n'
    write += 'Naive Bayes Classifier Accuracy:' + str(ac4) + '\n'
    write += 'Neural Network Classifier Accuracy:' + str(ac5) + '\n'
    write += 'GradientBoosting Classifier Accuracy:' + str(ac6) + '\n'
    write += 'Vote5 Classifier Accuracy:' + str(ac7) + '\n'
    write += ' ----------------------------------- \n'
    write += 'AdaBoost Classifier Report:\n' + str(rep1) + '\n'
    write += 'Random Forest Classifier Report:\n' + str(rep2) + '\n'
    write += 'K Nearest Neighbor Classifier Report:\n' + str(rep3) + '\n'
    write += 'Naive Bayes Classifier Report:\n' + str(rep4) + '\n'
    write += 'Neural Network Classifier Report:\n' + str(rep5) + '\n'
    write += 'GradientBoosting Report:\n' + str(rep6) + '\n'
    write += 'Vote5 Classifier Accuracy:\n' + str(rep7) + '\n'
    f.write(write)

with open("final result.txt", "w") as ff:
    write1 = 'Correct First Attempt\n'
    for i in range(len(bresult)):
        write1 += str(bresult[i]) + '\n'
    ff.write(write1)

    # 正负相关因子导致 结果overfit

final = pd.read_table('jch.csv')
final.drop('Correct First Attempt',axis= 1,inplace=True)
final.insert(11, 'Correct First Attempt', bresult)
final.to_csv('jjch.csv')