# Regression -- Linear Regression -- mean absolute percentage error

#step 1 : import library
import pandas as ps

    #step 2 : import data
salary = ps.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv")
salary.head()
salary.info()
salary.describe()
salary.shape

#step 3 : define dependet variable(y) and independet variable(x)
y = salary ['Salary']
x = salary [['Experience Years']]
salary.shape
x.shape
y.shape

    #step 4 : test train split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.3,random_state=2529)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

    #step 5 : select the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

'''  # notes
# error(e) = y(actual) - y(predicted)
if sum of (e)square = min---- then its the best fit line.
y = c + mx
salary = intercept + (slope * experience) + e'''

    # step 6 train model (fit model)
# fit - train data is used to predict
model.fit(x_train,y_train)
model.intercept_ #intercept
model.coef_ #slope

     #step 7 : prediction
y_prediction = model.predict(x_test) # only x is given in order to predict y

    #step 8 : accuracy
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_prediction)
