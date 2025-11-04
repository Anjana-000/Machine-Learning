import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv("https://raw.githubusercontent.com/jiss-sngce/CO_3/main/advertising.csv")
print(df.head())

sns.regplot(x="TV",y="Sales",data=df,line_kws={"color":"red"})
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

x_train,x_test,y_train,y_test=train_test_split(df[["TV"]],df[["Sales"]],test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)



print("coefficient=",model.coef_[0])
print("coefficient=",model.intercept_)

print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
