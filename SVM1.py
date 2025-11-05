import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Iris (1).csv")
print(df.head())

x=df.drop(["Id","Species"],axis=1)
y=df["Species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=47)

scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

svm_classifier=SVC(kernel='linear')
svm_classifier.fit(x_train,y_train)
y_pred=svm_classifier.predict(x_test)
print(classification_report(y_test,y_pred))
