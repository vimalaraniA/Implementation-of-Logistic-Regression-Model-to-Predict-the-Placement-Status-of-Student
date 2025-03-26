# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIMALA RANI A
RegisterNumber: 212223040240
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Placement_Data.csv") 
data.head()
```
![image](https://github.com/user-attachments/assets/ca0f6616-e1c9-4474-b221-0542ad1fb076)

```
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![image](https://github.com/user-attachments/assets/808a0582-0f2c-4b39-8fff-02639ebf1f44)

```
data1.isnull()
```
![image](https://github.com/user-attachments/assets/797771f6-a330-4ccc-94ef-4938ab863d99)

```
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/aa98c682-74eb-40be-99ce-cd1265a9dbe0)


```
le = LabelEncoder()
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in cols:
    data1[col] = le.fit_transform(data1[col])
data1
```
![image](https://github.com/user-attachments/assets/f502d7b6-d0c8-485a-98bc-1d78363df5f0)

```
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
```
![image](https://github.com/user-attachments/assets/96f6a06a-9411-4d52-99d2-3ec90168db93)

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/0fd052b3-68b5-429b-b889-21c49cdd943e)

```
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```
![image](https://github.com/user-attachments/assets/b00ae6d9-a3f0-43ff-82b5-8cc522f7b5b4)
```
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```
![image](https://github.com/user-attachments/assets/51c516b6-fc76-48cc-8cdf-2ea80f0f013a)
```
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
![image](https://github.com/user-attachments/assets/3e42bfff-006b-42bb-91e5-8d965fd99daf)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
