'''Decision Tree Classifier'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df=pd.read_csv("Iris (1).csv")
print(df.head())


x=df.drop(["Id","Species"],axis=1)
y=df["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=47)

model=DecisionTreeClassifier(criterion="entropy",min_samples_split=50)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

plt.figure(figsize=(5,10))
tree.plot_tree(model,filled=True,fontsize=10)
plt.show()

