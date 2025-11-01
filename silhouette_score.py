'''Program to implement k-means clustering technique using any standard dataset available in the public domain'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier,plot_tree

df=pd.read_csv("https://raw.githubusercontent.com/jiss-sngce/CO_3/main/jkcars.csv")
print(df.head())

print(df.shape)

new_data=df[['Volume','Weight','CO2']]
print(new_data.head())

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil_scores=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k,random_state=10,n_init=10)
    kmeans.fit(new_data)
    scores=silhouette_score(new_data,kmeans.labels_)
    sil_scores.append(scores)

silhouette_df=pd.DataFrame({'k':range(2,11),"silhouette_score":sil_scores})
print(silhouette_df)


plt.figure(figsize=(5,10))
plt.plot(silhouette_df['k'],silhouette_df['silhouette_score'])
plt.xlabel("K values")
plt.ylabel("silhouette score")
plt.show()

kmeans=KMeans(n_clusters=3,random_state=10)
kmeans.fit(new_data)

clusters=pd.Series(kmeans.predict(new_data))
df['Clusters']=clusters
print(df.head())
