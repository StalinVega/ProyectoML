from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
# EDA Packages



app = Flask(__name__)


def cargaRF():
    dataset = pd.read_csv('dataset.csv', delimiter=',', header=None)
    dataset=dataset.replace({"compliance": 1, "non-compliance ": 0})
    array = dataset.values
    X = array[:, 10:12]
    X = X.astype('int')
    y = array[:, 12]
    y = y.astype('int')
    X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=4)
    classes = {0:'non-compliance ',1:'compliance'}
    return X_train,X_test,y_train,y_test,classes

def cargaRL():
    dataset = pd.read_csv('dataset.csv', delimiter=',', header=None)
    dataset = dataset.replace({"compliance": 1, "non-compliance ": 0})
    array = dataset.values
    X = array[:, 10:12]
    X = X.astype('int')
    y = array[:, 12]
    y = y.astype('int')
    X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=4)
    classes = {0:'non-compliance ',1:'compliance'}
    return X_train,X_test,y_train,y_test,classes

def randRandom_Forest(X_train,X_test,y_train,y_test,classes,x_new):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print('------  randRandom_Forest  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracyRF= metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ',accuracyRF)
    y_predict=clf.predict(x_new)
    resultado=[]
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyRF,resultado




def Regresion_logistica(X_train,X_test,y_train,y_test,classes,x_new):
    algoritmo = LogisticRegression()
    algoritmo.fit(X_train, y_train)
    y_pred= algoritmo.predict(X_test)
    print('------  Regresion_logistica  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracyRL= metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ',accuracyRL)
    y_predict=algoritmo.predict(x_new)
    resultado=[]
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyRL,resultado





@app.route('/')
def index():
    return render_template("index.html")


@app.route('/randonF',methods=['POST'])
def nb():
    if request.method == 'POST':
        maxi=request.form['max1']
        mini=request.form['min1']
        x_new_rf = [[float(maxi), float(mini)]]
        X_train, X_test, y_train, y_test, classes = cargaRF()
        accuracyRF, resultadoRF = randRandom_Forest(X_train, X_test, y_train, y_test, classes, x_new_rf)

    return render_template("rf.html",accuracyRF=accuracyRF,resultadoRF=resultadoRF)


@app.route('/regresion',methods=['POST'])
def reglog():
    if request.method == 'POST':
        maxii=request.form['max']
        minii=request.form['min']
        x_new_rl = [[float(maxii), float(minii)]]
        X_train, X_test, y_train, y_test, classes = cargaRL()
        accuracyRL, resultadoRL = Regresion_logistica(X_train, X_test, y_train, y_test, classes, x_new_rl)

    return render_template("regresionL.html",accuracyRL=accuracyRL,resultadoRL=resultadoRL)


if __name__ == '__main__':
    app.run(port=5000,debug=True)