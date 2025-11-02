1 # Import all the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report

2 # Load the dataset
df=pd.read_csv("social-network-ads - social-network-ads.csv")
3# Print first five rows using head() function
print(df.head())

4 # Check if there are any null values. If any column has null values, treat them accordingly
print(df.isnull().sum())

1 # Split the dataset into dependent and independent features
x=df.drop(["User ID","Purchased"],axis=1)
y=df["Purchased"]

# Use &#39;info()&#39; function with the features DataFrame.
print(df.info())
# Use get_dummies()&#39; function to convert each categorical column in a DataFrame to numerical.
x=pd.get_dummies(df,columns=["Gender"])
print(x)

 # Split the DataFrame into the train and test sets.
 # Perform train-test split using &#39;train_test_split&#39; function.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=47)

# Print the shape of the train and test sets.
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

#Train kNN Classifier model
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

# Perform prediction using &#39;predict()&#39; function.
y_pred=model.predict(x_test)

# Call the &#39;score()&#39; function to check the accuracy score of the train set and test set.
score=model.score(x_train,y_train)
print(score)
score1=model.score(x_test,y_test)
print(score1)

# Display the precision, recall, and f1-score values.
print("Precision=",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("f1 score:",f1_score(y_test,y_pred))
