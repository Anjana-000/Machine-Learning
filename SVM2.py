data = [
    "Apple launched a new iPhone with better neural engine.",  # tech
    "The stock market saw huge gains after the quarterly report.", # finance
    "Google's machine learning model achieved 90% accuracy.",  # tech
    "Investors are worried about rising interest rates and inflation.", # finance
    "Python libraries like scikit-learn are great for ML.", # tech
    "Bonds and treasury yields are highly volatile this week." # finance
]
lables=[0,1,0,1,0,1]

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

x_train,x_test,y_train,y_test=train_test_split(data,lables,test_size=0.33,random_state=47)

vectorizer=TfidfVectorizer()
x_train_tfidf=vectorizer.fit_transform(x_train)
x_test_vectors=vectorizer.transform(x_test)

svm_classifier=SVC(kernel="linear")
svm_classifier.fit(x_train_tfidf,y_train)
y_pred=svm_classifier.predict(x_test_vectors)

new_text=["Bonds and treasury yields are highly volatile this week."]
new_text_vec=vectorizer.transform(new_text)
prediction=svm_classifier.predict(new_text_vec)

if prediction[0]==0:
    print(f"this text'{new_text[0]}' is tech")
else:
    print(f"this text'{new_text[0]}' is finance")

