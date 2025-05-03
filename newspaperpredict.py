import pandas as pd

true=pd.read_csv("C:/Users/sures/Desktop/news Predection/news/True.csv")
fake=pd.read_csv("C:/Users/sures/Desktop/news Predection/news/Fake.csv")
true['label']='T'
fake['label']='F'
true
fake
news = pd.concat([true, fake])

news=news.drop(['title','subject','date'],axis=1,errors='ignore')
news

news=news.sample(frac=1)
news.reset_index(inplace=True)
news.drop(['index'], axis=1, inplace=True)
news

import re

def wordop(text):
  text = text.lower()
  text = re.sub('<.*?>', '', text)  # Remove HTML tags
  text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
  text = re.sub('\n', '', text)  # Remove newline characters
  text = re.sub('[^\w\s]+', '', text)  # Remove special characters
  text = re.sub('\d', ' ', text)  # Replace digits with spaces
  text = re.sub('\s+', ' ', text).strip()  # Remove extra spaces
  return text

news['text'] = news['text'].apply(wordop)
news

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(news['text'], news['label'], test_size=0.3)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

vector=TfidfVectorizer()
xv_train=vector.fit_transform(x_train)
xv_test=vector.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

LR=LogisticRegression()
LR.fit(xv_train,y_train)
pred=LR.predict(xv_test)

LR.score(xv_test,y_test)
print(classification_report(y_test,pred))

vector=TfidfVectorizer()
xv_train=vector.fit_transform(x_train)
xv_test=vector.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred=DT.predict(xv_test)
DT.score(xv_test,y_test)
print(classification_report(y_test,pred))

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier()
RF.fit(xv_train,y_train)
pred=RF.predict(xv_test)
RF.score(xv_test,y_test)
print(classification_report(y_test,pred))

def output(n):
  if n == 0:
    return "it is Fake News"
  elif n == 1:
    return "it is Genuine News"

def writen(news):
  testnews={"text":[news]}
  new=pd.DataFrame(testnews)
  new["text"]=new["text"].apply(wordop)
  newx_test=new["text"]
  newxv_test=vector.transform(newx_test)
  pred=LR.predict(newxv_test)
  pred_d = DT.predict(newxv_test)[0]
  pred_R = RF.predict(newxv_test)[0]
  pred_d = 0 if pred_d == 'F' else 1
  pred_R = 0 if pred_R == 'F' else 1
  return "DT Prediction:{}\nRF Prediction:{}".format(output(pred_d),output(pred_R))

news=str(input())
pred = writen(news)
print(pred)