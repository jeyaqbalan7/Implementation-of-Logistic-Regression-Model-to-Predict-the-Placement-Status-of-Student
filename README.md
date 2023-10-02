# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries which are used for the program.

2.Load the dataset.

3.Check for null data values and duplicate data values in the dataframe.

4.Apply logistic regression and predict the y output.

5.Calculate the confusion,accuracy and classification of the dataset.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Jeyabalan
RegisterNumber:  212222240040

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/eafd6d5d-f121-4ff0-8fa4-b01afdda7e02)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/913db7d6-2eee-4f6d-8983-9cd67123bd50)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/3db6c85c-84b7-49e8-806e-f4e541d15246)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/958fd9c5-3205-40fa-bcf7-847c73b62d2d)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/7517463e-327a-42c7-96ee-94b12871ef3a)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/121fbef3-c2f0-456d-8435-71b9bf77de7b)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/60f86768-926e-4690-9f8f-53cb46d60597)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/eb92e72e-8dc5-4d0d-8d19-3f37d4ede062)

![image](https://github.com/jeyaqbalan7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393851/de50a2e0-3f01-4a50-bc2a-f1c9b122e074)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
