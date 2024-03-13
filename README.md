# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student->

## AIM :
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## PROGRAM :
#### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#### Developed by : AJAY ASWIN M
#### RegisterNumber : 212222240005
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## OUTPUT:

### Placement_data
![311776280-2d30f6e7-146a-4759-b90e-6e50676a2bc8](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/f6382a72-2857-4520-81a2-cc7f4a5c1727)

### Salary_data
![311776407-0d52c60f-4712-4e31-bdac-8ab2b3229c47](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/7073f537-bf11-4aaa-9b46-99124bbf7788)

### ISNULL()
![311776510-b9f45afa-0d93-421f-9d73-d3fc072250e6](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/106b184c-99de-4034-9e19-e1288ea9bde2)


### DUPLICATED()
![311776804-dcf0bf6e-543f-41d2-82f8-2cb93777173e](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/9ee083df-488e-413b-9d35-4541ddeeba0f)

### Print Data
![311777071-4659883d-f53a-4d4d-8183-8b36cd480d45](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/a8095bec-d5c2-4218-9fb4-ef73f38296e3)

### iloc[:,:-1]
![311777668-dd8c3742-8749-4cac-ad58-cdf86433b80b](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/51931693-b169-4ad4-a274-8863ec0bba38)

### Data_Status
![311777821-af901a2d-25fe-493e-9d6c-ec33cbf23e90](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/b57bcf7b-c13d-4d71-a6f2-275cc9d38f4a)

### Y_Prediction array:
![311778090-50a32c48-d67d-49ab-9e0c-20140d857436-1](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/e1b2f5a3-8f84-42a0-b84e-574c23aa12d3)

### Accuray value:
![311778202-77b946f3-a6b2-4a98-8566-ca6878cacad6](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/2b941d2c-b078-4b80-b578-b4f784a37396)

### Confusion Array:
![311778303-7edbb883-e626-428b-9648-e8726cb252e5](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/14f7a044-e6d5-4e25-9a5e-f39f047acc51)

### Classification report:
![311778375-05e10820-342d-4c64-b51d-c73795e46d92](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/b437ac2d-474a-4d6f-906d-50d3f544e829)

### Prediction of LR:
![311775722-05562708-e9ca-43a3-8981-a5834e4a2948](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/66b14f82-486b-43dd-9cdb-3d672757dcaf)


## RESULT :
Thus,the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
