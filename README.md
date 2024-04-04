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
```python
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

### 1. Placement_data
![311776280-2d30f6e7-146a-4759-b90e-6e50676a2bc8](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/6ad376df-02e7-4d13-ba37-645ff36c5f06)


### 2. Salary_data
![266196827-0dd174c9-5101-44ef-bc5e-c6d171f3cec2](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/5650c072-bb0f-4ae9-9b23-96d64b01e83c)

### 3. ISNULL()
![311776510-b9f45afa-0d93-421f-9d73-d3fc072250e6](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/58df6db6-79be-4a87-8dfe-aa3b7c303edc)

### 4. DUPLICATED()
![311776804-dcf0bf6e-543f-41d2-82f8-2cb93777173e](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/904fe17f-1884-49d9-9923-df9ec03f5f6d)

### 5. Print Data
![311777071-4659883d-f53a-4d4d-8183-8b36cd480d45](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/616dbc71-3349-4905-aaf1-6601cd5dd536)

### 6.iloc[:,:-1]
![311777668-dd8c3742-8749-4cac-ad58-cdf86433b80b](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/2177ba47-4fd8-4947-b522-4a044e2ae41c)

### 7.Data_Status
![311777821-af901a2d-25fe-493e-9d6c-ec33cbf23e90](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/1f870de8-830d-4d76-8e30-ac7fa2937099)

### 8.Y_Prediction array:
![311778090-50a32c48-d67d-49ab-9e0c-20140d857436](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/74ac4f1b-c69d-4fa1-a22c-931031a11b17)

### 9.Accuray value:
![311778202-77b946f3-a6b2-4a98-8566-ca6878cacad6](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/922470f0-359a-4084-8a9f-d296a151eba4)

### 10.Confusion Array:
![311778303-7edbb883-e626-428b-9648-e8726cb252e5](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/936fada0-ca40-4068-82e1-07034c1711e8)

### 11. Classification report:
![311778375-05e10820-342d-4c64-b51d-c73795e46d92](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/66acbc38-f209-4029-aaa1-db1f0e1afbc8)

### 12. Prediction of LR:
![311775722-05562708-e9ca-43a3-8981-a5834e4a2948](https://github.com/AJAYASWIN-M/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679692/7955cc32-eb14-4004-b193-9635bc0ca53d)


## RESULT :
Thus,the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
