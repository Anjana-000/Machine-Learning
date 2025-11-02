'''Aim: Program to implement Naïve Bayes Algorithm using any standard dataset available in the public domain and find the accuracy of the
algorithm
✓ Short notes: Naive Bayes
Bayes&#39; Theorem provides a way that we can calculate the probability of a piece of data belonging to a given class, given
our prior knowledge. Bayes&#39; Theorem is stated as:
P(class data) = (P(data|class) * P(class)) / P(data)
Where P(class data) is the probability of class given the provided data.
We are using Iris Dataset. The Iris Flower Dataset involves predicting the flower species given measurements of iris flowers.
It is a multiclass classification problem. The number of observations for each class is balanced. There are 150 observations with 4 input
variables and 1 output variable. The variable names are as follows:
Sepal length in cm.
Sepal width in cm.
Petal length in cm.
Petal width in cm.
Class.
Algorithm:
Step 1: Separate By
Class.
Step 2: Summarize Dataset.
Step 3: Summarize Data By
Class.
Step 4: Gaussian Probability Density Function.
Step 5: Class
Probabilities.'''



import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("Iris (1).csv")
print(df.head())

x=df.drop(["Id","Species"],axis=1)
y=df["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=47)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Precision:",classification_report(y_test,y_pred))
