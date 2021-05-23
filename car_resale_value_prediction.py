# Importing required libraries
import pickle
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Importing the dataset
df = pd.read_csv('car_data.csv')
print(df.head())

# Checking the dataframe shape
print(df.shape)

# Exploring data statistics
print(df.describe())

# Checking for any null values
print(df.isnull().sum())

# Printing all categorical features and their values
print('Types of fuel:', df['Fuel_Type'].unique())
print('Types of seller:', df['Seller_Type'].unique())
print('Types of transmission:', df['Transmission'].unique())
print('Types of owner:', df['Owner'].unique())

# Dropping unrequired features from dataframe
df.drop(['Car_Name'], axis = 1, inplace = True)

# Creating new feature - 'num_years' (current year - year) to calculate age of car
df['Num_Years'] = datetime.datetime.now().year - df['Year']

# Dropping feature 'year' (car manufacture year) from dataframe
df.drop(['Year'], axis = 1, inplace = True)

# Converting categorical features into dummy variables
df = pd.get_dummies(df, drop_first = True)

# Getting correlation matrix for the dataset
print(df.corr())

# Creating a Resale Value Percentage feature for understanding data
df['Resale_Percentage'] = round(df['Selling_Price'] / df['Present_Price'] * 100, 2)

# Plotting the pairplot for the dataset
sns.pairplot(df[['Selling_Price', 'Present_Price', 'Kms_Driven', 'Num_Years', 'Resale_Percentage']])
plt.show()

# Plotting the correlation heatmap
plt.figure(figsize = (20, 20))
sns.heatmap(df.corr(), annot = True, cmap = 'RdBu')
plt.show()

# Dropping the 'Resale_Percentage' feature
df.drop(['Resale_Percentage'], axis = 1, inplace = True)

# Extracting dependent and independent features
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Getting feature importances
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)

# Plotting barplot feature importances
feature_imp = pd.Series(model.feature_importances_, index = X.columns)
feature_imp.plot(kind = 'barh')
plt.show()

# Dropping 'Owner' feature as its importance is very low
df.drop(['Owner'], axis = 1, inplace = True)

# Extracting features from updated dataframe
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Getting new feature importances
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)

# Plotting barplot for new feature importances
feature_imp = pd.Series(model.feature_importances_, index = X.columns)
feature_imp.plot(kind = 'barh')
plt.show()

# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)

# Creating a linear regression model
linear_regressor = LinearRegression(n_jobs=-1)
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)

# Computing some performance metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Plotting histogram of difference between y_pred and y_test
sns.displot(y_test - y_pred, kind = 'kde')
plt.xlabel('Selling Price (y_test - y_pred)')
plt.show()

# Plotting scatterplot between y_pred and y_test
plt.scatter(y_test, y_pred)
plt.title('True Value vs Predicted Value')
plt.ylabel('y_test')
plt.xlabel('y_pred')
plt.show()

# Creating a hyperparamter value grid for hyperparamter tuning
random_grid = {
    'C' : [0.1, 1, 10],
    'epsilon' : [0.01, 0.1, 1]
}

# Creating a RandomSearchCV model
support_vector_regressor = RandomizedSearchCV(estimator=SVR(), param_distributions=random_grid,
                                              scoring='neg_mean_squared_error', n_iter=10, cv=5, 
                                              verbose=1, n_jobs=-1)
support_vector_regressor.fit(X_train, y_train)

# Printing best score and best hyperparameter values
print('Best Score:', support_vector_regressor.best_score_)
print('Best Hyperparamters:', support_vector_regressor.best_params_)

# Creating final model with best hyperparameter values
support_vector_regressor = SVR(C=10, epsilon=1)
support_vector_regressor.fit(X_train, y_train)
y_pred = support_vector_regressor.predict(X_test)

# Computing some performance metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Plotting histogram of difference between y_pred and y_test
sns.displot(y_test - y_pred, kind = 'kde')
plt.xlabel('Selling Price (y_test - y_pred)')
plt.show()

# Plotting scatterplot between y_pred and y_test
plt.scatter(y_test, y_pred)
plt.title('True Value vs Predicted Value')
plt.ylabel('y_test')
plt.xlabel('y_pred')
plt.show()

# Creating a default (simple) Random Forest Regressor model
random_forest_regressor = RandomForestRegressor(n_jobs=-1)
random_forest_regressor.fit(X_train, y_train)
y_pred = random_forest_regressor.predict(X_test)

# Performance Metrics
print('R2 score: ', r2_score(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))

# Creating RandomizedSearchCV grid

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Creating a RandomSearchCV model
random_forest_regressor = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
                                             scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=1, n_jobs=-1)
random_forest_regressor.fit(X_train, y_train)

# Printing the best parametets and best score
print('Best Parameter:', random_forest_regressor.best_params_)
print('Best Score:', random_forest_regressor.best_score_)

# Creating a final random Forest Regression model with best hyperparameter values
random_forest_regressor = RandomForestRegressor(n_estimators=1000, min_samples_split=2, min_samples_leaf=1,
                                                max_features='auto', max_depth=20, n_jobs=-1)
random_forest_regressor.fit(X_train, y_train)
y_pred = random_forest_regressor.predict(X_test)

# Computing some performance metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Plotting histogram of difference between y_pred and y_test
sns.displot(y_test - y_pred, kind = 'kde')
plt.xlabel('Selling Price (y_test - y_pred)')
plt.show()

# Plotting scatterplot between y_pred and y_test
plt.scatter(y_test, y_pred)
plt.title('True Value vs Predicted Value')
plt.ylabel('y_test')
plt.xlabel('y_pred')
plt.show()

# Exporting the models to a pickle file

# Open file in desired location/directory
file = open('models.pkl', 'wb') 
# Dump information to that file
pickle.dump([linear_regressor, support_vector_regressor, random_forest_regressor], file)
# Close the file
file.close()