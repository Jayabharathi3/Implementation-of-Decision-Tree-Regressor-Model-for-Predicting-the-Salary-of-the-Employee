# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JAYABHARATHI S
RegisterNumber: 212222100013  
*/
```
``` python

import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])


```

## Output:

## INITIAL DATASET
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/3d75eb95-5d88-4f92-9f82-b05db357e6b5)

## DATA.INFO ()
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/4ce86a6f-c2ec-4be2-b046-13148752b782)

## OPTIMIZATION OF NULL VALUES
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/e8cdaf67-5163-437d-bd2c-4f1e07e52ee9)

## Converting string literals to numerical values using label encoder:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/b1333c5c-6329-491e-960c-3c6170cadc4a)

## MSE Value
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/217832af-8cb3-4618-ac2d-56a18454acd2)

## R2 VALUE (VARIANCE) :
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/548469a1-d792-4d48-8233-991e8fc1d7c0)

## DATA PREDICTION
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120367796/2e877b04-fafe-4c10-a3bb-79d84e20a780)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
