#import lib
import pandas as pd

#import data
dai = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes%20Missing%20Data.csv')
dai.columns

#define x and y 
y = dai['Class']
x = dai[['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin',
       'BMI', 'Diabetes_Pedigree', 'Age']]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

#select model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=500)

#fit model
model.fit(x_train,y_train)

#predict
y_pred = model.predict(x_test)

#accuracy
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)