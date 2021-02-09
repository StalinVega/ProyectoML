from flask import Flask, render_template
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# EDA Packages



app = Flask(__name__)

def Naive_Bayes(X_train,X_test,y_train,y_test,classes,x_new):
    #random_state=42
    #X_train,X_test,y_train,y_test,classes=cargariris(random_state)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print('------  Naive_Bayes  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracyNB= metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ',accuracyNB)
    y_predict=gnb.predict(x_new)
    resultado=[]
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyNB,resultado


def Regresion_logistica(X_train,X_test,y_train,y_test,classes,x_new):
    #random_state=None
    #X_train,X_test,y_train,y_test,classes=cargariris(random_state)
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


if __name__ == '__main__':
    app.run(port=5000,debug=True)