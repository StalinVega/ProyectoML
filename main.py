from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
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

def randRandom_Forest(X_train,X_test,y_train,y_test,classes,x_new):
    #random_state=42
    #X_train,X_test,y_train,y_test,classes=cargariris(random_state)
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




def Regresion_logistica(x_new):
    # Importar el dataset
    df = pd.read_csv('dataset.csv', delimiter=',', header=None)
    # Ver los 5 primeros valores del dataset
    # Identificar en el conjunto de datos las variables independientes y dependientes
    array = df.values
    X = array[:, 10:12]
    y = array[:, 12]
    # División del conjunto de datos en el conjunto de entrenamiento y el conjunto de prueba
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=82)

    # Escala de características para llevar la variable en una sola escala
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Adaptación de la clasificación logística multiclase al conjunto de entrenamiento
    from sklearn.linear_model import LogisticRegression
    logisticregression = LogisticRegression()
    logisticregression.fit(X_train, y_train)

    # Predecir los resultados del conjunto de pruebas
    y_pred = logisticregression.predict(X_test)
    print(y_pred)

    # lets ven el valor real y el pronosticado uno al lado del otro
    y_compare = np.vstack((y_test, y_pred)).T

    # valor real en el lado izquierdo y valor predicho en el lado derecho
    # impresión de los 5 valores principales
    y_compare[:5, :]

    # Matriz de confusion
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # precisión de búsqueda de la matriz de confusión.
    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred += cm[row, c]
            else:
                falsePred += cm[row, c]
    acurry = corrPred / (cm.sum())

    print('Correct predictions: ', corrPred)
    print('False predictions', falsePred)
    print('Accuracy of the multiclass logistic classification is: ', corrPred / (cm.sum()))

    return acurry,





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


@app.route('/regresion')
def reglog():
    return render_template("regresionL.html")


if __name__ == '__main__':
    app.run(port=5000,debug=True)