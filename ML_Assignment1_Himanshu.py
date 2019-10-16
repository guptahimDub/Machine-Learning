
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer

dataset=pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
indexNames2 = dataset[ (dataset['Income in EUR'] < 0)].index
dataset.drop(indexNames2 , inplace=True)

simpleimputermedian=SimpleImputer(strategy='median')
dataset['Year of Record']=simpleimputermedian.fit_transform(dataset['Year of Record'].values.reshape(-1,1))
dataset['Age']=simpleimputermedian.fit_transform(dataset['Age'].values.reshape(-1,1))
dataset['Body Height [cm]']=simpleimputermedian.fit_transform(dataset['Body Height [cm]'].values.reshape(-1,1))
datasetnoncateg=dataset.drop(['Instance','Hair Color','Wears Glasses','Hair Color'],axis=1)

dataWOlabel=pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

dataWOlabel['Year of Record']=simpleimputermedian.fit_transform(dataWOlabel['Year of Record'].values.reshape(-1,1))
dataWOlabel['Age']=simpleimputermedian.fit_transform(dataWOlabel['Age'].values.reshape(-1,1))
dataWOlabel['Body Height [cm]']=simpleimputermedian.fit_transform(dataWOlabel['Body Height [cm]'].values.reshape(-1,1))
Mnoncateg=dataWOlabel.drop(['Instance','Hair Color','Wears Glasses','Hair Color','Income'],axis=1)

X=datasetnoncateg.drop('Income in EUR',axis=1).values
Y=datasetnoncateg['Income in EUR'].values

t1 = TargetEncoder()
t1.fit(X, Y)
X = t1.transform(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=0)

regressor = RandomForestRegressor()


fitResult = regressor.fit(Xtrain, Ytrain)
YPredTest = regressor.predict(Xtest)

np.sqrt(metrics.mean_squared_error(Ytest, YPredTest))


A=Mnoncateg.values
A=t1.transform(A)
B=regressor.predict(A)

df2=pd.DataFrame()
df2['Instance']=M['Instance']
df2['Income']=B

df2.to_csv('ML_Assignment1_Output8.csv',index=False)

