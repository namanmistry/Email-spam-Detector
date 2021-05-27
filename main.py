import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import dump, load

class Email:
    def __init__(self, text, label):
        self.text = text
        self.label = label

Emails = []

count = 0
for i in os.listdir("./data/ham"):
    if count>=1500:
        break
    count+=1
    with open(f"./data/ham/{i}") as f:
        text = f.read()
        Emails.append(Email(text,1))

for i in os.listdir("./data/spam"):
    with open(f"./data/spam/{i}", encoding="utf8",errors='ignore') as f:
        text = f.read()
        Emails.append(Email(text,0))

X = [email.text for email in Emails]
y = [email.label for email in Emails]

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_tr = vectorizer.fit_transform(X_train)
x_test_tr = vectorizer.transform(x_test)

nm = MultinomialNB()
nm.fit(X_train_tr, Y_train)

pred = nm.predict(x_test_tr)

mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(mse)

dump(nm, "./model/EmailSpamDetector.joblib")