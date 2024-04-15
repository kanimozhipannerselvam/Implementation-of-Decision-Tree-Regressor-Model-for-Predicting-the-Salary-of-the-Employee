# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KANIMOZHI
RegisterNumber:  212222230060
*/
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

## Output:
#HEAD(), INFO() & NULL():

![318907613-ebe4c962-cd19-4d77-aa16-2597409884cd](https://github.com/kanimozhipannerselvam/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119476060/f957dc8e-65fd-4bd7-be34-06729560f174)

#Converting string literals to numerical values using label encoder:

![318907957-5ae14c42-9b16-4a5a-a20f-81961ca44642](https://github.com/kanimozhipannerselvam/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119476060/cf728228-fb3e-4e8a-b91a-7d3b9cf53d87)

#MEAN SQUARED ERROR:

![318908213-3aaad08c-7b33-45af-b78b-3ef21be92cf7](https://github.com/kanimozhipannerselvam/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119476060/97004818-5dc4-479b-b91c-719fa8003cb9)

#R2 (Variance):

![318908472-d1032744-0b03-4b1e-b3f5-11711de5f180](https://github.com/kanimozhipannerselvam/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119476060/dde34d51-05c3-497c-88e5-507f8db4b530)

#DATA PREDICTION & DECISION TREE REGRESSOR FOR PREDICTING THE SALARY OF THE EMPLOYEE

![318909073-92b71892-3ff3-49db-9864-69ef5271d8fa](https://github.com/kanimozhipannerselvam/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119476060/4ab7ab73-182c-4379-8c5c-de2f9bb92dbf)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
