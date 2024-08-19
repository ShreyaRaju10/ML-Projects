#import library
import pandas as pd

#import data
mpg = pd.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv")
mpg.head()
mpg.info()
mpg.describe()
mpg.nunique()

#column's title
mpg.columns

#define x and y
y = mpg['mpg']
x = mpg[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model_year', 'origin', 'name']]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)

# Separate string and int columns
int_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
str_cols = ['origin', 'name']

#SimpleImputer -- handles int column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') #replaces missing values with the mean of each column.
x_train[int_cols] = imputer.fit_transform(x_train[int_cols])
x_test[int_cols] = imputer.transform(x_test[int_cols])

#OneHotEncoder -- handles str column
from sklearn.preprocessing import OneHotEncoder 
encoder = OneHotEncoder(handle_unknown='ignore') # Creates a binary column for each category
x_train_encoded = encoder.fit_transform(x_train[str_cols])
x_test_encoded = encoder.transform(x_test[str_cols])


# Convert encoded data to arrays and concatenate with numeric data
import numpy as np
a = x_train[int_cols].values #access the values of the column
b = x_train_encoded.toarray() #converts string data to array
c = x_test[int_cols].values
d= x_test_encoded.toarray()
x_train_final = np.concatenate((x_train[int_cols].values,x_train_encoded.toarray()),axis=1) #axis=1 concatenation along column, =0 along row
x_test_final = np.concatenate((x_test[int_cols].values,x_test_encoded.toarray()),axis=1)

#select model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()

#fit model
model.fit(x_train_final,y_train)

#predict 
y_pred = model.predict(x_test_final)
y_pred

#accuracy
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)

#data visualisation 
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred_2,color = 'hotpink' ,s=70, alpha=0.5) #alpha--transparency 
plt.plot(y_test,y_test,color = '#88c999')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')
plt.show()
