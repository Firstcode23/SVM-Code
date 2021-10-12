import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


def read_data():
    data = pd.read_csv('C:\\Sudarshan\\ds_training\\Datasets\\SVM_Classification_Data.csv')
    return data



df = read_data()
print(df.head(5))
print(df.shape)
print(df.size)
print(df.dtypes)
print(df.columns)
print(df.isnull().sum())
X = df.drop('diabetes', axis = 1)
y = df['diabetes']
print(X.shape,y.shape)
stand_scaler = StandardScaler()
X_scaleddata = stand_scaler.fit_transform(X)
print(X_scaleddata)
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train,y_train)
y_predict = svm_model.predict(X_test)
print(metrics.accuracy_score(y_test,y_predict))
print(metrics.precision_score(y_test,y_predict))
print(metrics.recall_score(y_test,y_predict))