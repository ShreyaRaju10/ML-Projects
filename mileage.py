from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.impute import SimpleImputer # Import SimpleImputer to handle NaNs

# Assuming 'mpg' is your DataFrame
mpg = mpg.dropna() # Drop rows with missing values 

y = mpg['mpg']
x = mpg[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

model = LinearRegression()
model_1 = HistGradientBoostingClassifier()

model.fit(x_train,y_train)
# Note: HistGradientBoostingClassifier can handle NaNs natively, so no imputation is needed for model_1
model_1.fit(x,y)




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer # to handle NaNs
from sklearn.preprocessing import OneHotEncoder # to handle categorical data
mpg = mpg.dropna() # Drop rows with missing values

y = mpg['mpg']
x = mpg[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model_year', 'origin', 'name']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)

# Separate numeric and categorical columns
numeric_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
categorical_cols = ['origin', 'name']

# Impute missing values in numeric columns using the mean strategy
imputer = SimpleImputer(strategy='mean') #replaces missing values with the mean of each column.
x_train[numeric_cols] = imputer.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = imputer.transform(x_test[numeric_cols])

# Handle categorical columns using OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore') # Creates a binary column for each category
x_train_encoded = encoder.fit_transform(x_train[categorical_cols])
x_test_encoded = encoder.transform(x_test[categorical_cols])

# Convert encoded data to arrays and concatenate with numeric data
x_train_final = np.concatenate((x_train[numeric_cols].values, x_train_encoded.toarray()), axis=1)
x_test_final = np.concatenate((x_test[numeric_cols].values, x_test_encoded.toarray()), axis=1)

model = LinearRegression()
model_1 = HistGradientBoostingRegressor()

model.fit(x_train_final,y_train)
# model_1.fit(x,y) # This line will still cause errors, you need to preprocess 'x' similarly