import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold


# Function for splitting strings into characters
def split_column_by_characters(df, col_name):
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' does not exist in the DataFrame.")
    if not all(isinstance(val, str) for val in df[col_name]):
        raise ValueError(f"Column '{col_name}' must contain only strings.")
    new_cols = df[col_name].apply(list).tolist()
    max_length = max(len(chars) for chars in new_cols)
    new_col_names = [f'{col_name}_{i+1}' for i in range(max_length)]
    new_df = pd.DataFrame(new_cols, columns=new_col_names).fillna('')
    return pd.concat([df, new_df], axis=1)


#load el data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.drop(['X1','X2','X3','X5','X8','X10'], axis=1)
test = test.drop(['X1','X2','X3','X5','X8','X10'], axis=1)


# Handle missing values
for df in [train, test]:
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])  # Replace missing values with mode
        else:
            df[column] = df[column].fillna(df[column].median())  # Replace missing values with median




# Removing Numerical Outliers from Training Dataset
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns

# Check if 'X4' is in the numeric columns and exclude it
if 'X4' in numeric_cols:
    numeric_cols = numeric_cols.drop('X4')  # X4 has high skewness; Z-score method is not suitable for it

# Detect and replace outliers for the remaining numeric columns using Z-score
z_scores = train[numeric_cols].apply(zscore)
threshold = 3
outliers = (z_scores.abs() > threshold).any(axis=1)

for column in numeric_cols:
    median = train[column].median()
    train.loc[outliers, column] = median

print(f"Replaced outliers in numerical columns of training data with their median values using Z-score.")

# Handle outliers in 'X4' using the IQR method
Q1 = train['X4'].quantile(0.25)
Q3 = train['X4'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

X4_outliers = (train['X4'] < lower_bound) | (train['X4'] > upper_bound)
train.loc[X4_outliers, 'X4'] = train['X4'].median()

print(f"Removed outliers from 'X4' using the IQR method.")



# Repeat the same logic for the test dataset
numeric_cols_test = test.select_dtypes(include=['float64', 'int64']).columns

if 'X4' in numeric_cols_test:
    numeric_cols_test = numeric_cols_test.drop('X4')  # Exclude X4 for the same reason as training

# Detect and replace outliers in the test dataset using Z-score
z_scores_test = test[numeric_cols_test].apply(zscore)
outliers_test = (z_scores_test.abs() > threshold).any(axis=1)

for column in numeric_cols_test:
    median = test[column].median()
    test.loc[outliers_test, column] = median

print(f"Replaced outliers in numerical columns of test data with their median values using Z-score.")

# Handle outliers in 'X4' using the IQR method for the test dataset
Q1_test = test['X4'].quantile(0.25)
Q3_test = test['X4'].quantile(0.75)
IQR_test = Q3_test - Q1_test
lower_bound_test = Q1_test - 1.5 * IQR_test
upper_bound_test = Q3_test + 1.5 * IQR_test

X4_outliers_test = (test['X4'] < lower_bound_test) | (test['X4'] > upper_bound_test)
test.loc[X4_outliers_test, 'X4'] = test['X4'].median()

print(f"Removed outliers from 'X4' in test data using the IQR method.")



# One-hot encode remaining categorical variables
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)



# Standardize the numerical data
Train_Numerical_columns = train.select_dtypes(include=['float64', 'int64']).columns.drop('Y')
Test_Numerical_columns = test.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler() #to standardize el data
train[Train_Numerical_columns] = scaler.fit_transform(train[Train_Numerical_columns])
test[Test_Numerical_columns] = scaler.transform(test[Test_Numerical_columns])

# Split the train dataset
X = train.drop('Y', axis=1)
y = train['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #splitting data to a 70 30

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False) #second to not overfit///including the bias is redundant///transformer objects  finds polynomial combinations
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
test_poly = poly.transform(test)

# Train Polynomial Regression Model
model = LinearRegression()
model.fit(X_train_poly, y_train) #finds the poly equation

# Predictions and evaluation
error_prediction = model.predict(X_test_poly) #predicts the values(Y) from the calculated equation->series
mae = mean_absolute_error(y_test, error_prediction) #finds mean absolute error between the actual Ys and the predicted Ys to test before submitting
predictions = model.predict(test_poly) #predicts the required prices

print("Mean Absolute Error:", mae)

# Save predictions
output_test = pd.DataFrame({ #creates a table of the ids and the predicted Ys
    'row_id': range(len(predictions)),
    'Y': predictions
})
output_test.to_csv("sample_submission.csv", index=False ) #overwrites the sample submission file
