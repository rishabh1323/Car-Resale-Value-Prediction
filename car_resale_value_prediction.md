# Car Resale Value Prediction

## Kaggle Vehicle Dataset
**[Link to Dataset](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=CAR+DETAILS+FROM+CAR+DEKHO.csv)**


```python
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

%matplotlib inline
```


```python
# Importing the dataset
df = pd.read_csv('car_data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Car_Name</th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ritz</td>
      <td>2014</td>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sx4</td>
      <td>2013</td>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ciaz</td>
      <td>2017</td>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wagon r</td>
      <td>2011</td>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>swift</td>
      <td>2014</td>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the dataframe shape
df.shape
```




    (301, 9)




```python
# Exploring data statistics
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>301.000000</td>
      <td>301.000000</td>
      <td>301.000000</td>
      <td>301.000000</td>
      <td>301.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2013.627907</td>
      <td>4.661296</td>
      <td>7.628472</td>
      <td>36947.205980</td>
      <td>0.043189</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.891554</td>
      <td>5.082812</td>
      <td>8.644115</td>
      <td>38886.883882</td>
      <td>0.247915</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2003.000000</td>
      <td>0.100000</td>
      <td>0.320000</td>
      <td>500.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2012.000000</td>
      <td>0.900000</td>
      <td>1.200000</td>
      <td>15000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.000000</td>
      <td>3.600000</td>
      <td>6.400000</td>
      <td>32000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.000000</td>
      <td>6.000000</td>
      <td>9.900000</td>
      <td>48767.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2018.000000</td>
      <td>35.000000</td>
      <td>92.600000</td>
      <td>500000.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking for any null values
df.isnull().sum()
```




    Car_Name         0
    Year             0
    Selling_Price    0
    Present_Price    0
    Kms_Driven       0
    Fuel_Type        0
    Seller_Type      0
    Transmission     0
    Owner            0
    dtype: int64




```python
# Printing all categorical features and their values
print('Types of fuel:', df['Fuel_Type'].unique())
print('Types of seller:', df['Seller_Type'].unique())
print('Types of transmission:', df['Transmission'].unique())
print('Types of owner:', df['Owner'].unique())
```

    Types of fuel: ['Petrol' 'Diesel' 'CNG']
    Types of seller: ['Dealer' 'Individual']
    Types of transmission: ['Manual' 'Automatic']
    Types of owner: [0 1 3]
    


```python
# Dropping unrequired features from dataframe
df.drop(['Car_Name'], axis = 1, inplace = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013</td>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating new feature - 'num_years' (current year - year) to calculate age of car
df['Num_Years'] = datetime.datetime.now().year - df['Year']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
      <th>Num_Years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013</td>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dropping feature 'year' (car manufacture year) from dataframe
df.drop(['Year'], axis = 1, inplace = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
      <th>Num_Years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Converting categorical features into dummy variables
df = pd.get_dummies(df, drop_first = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Owner</th>
      <th>Num_Years</th>
      <th>Fuel_Type_Diesel</th>
      <th>Fuel_Type_Petrol</th>
      <th>Seller_Type_Individual</th>
      <th>Transmission_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting correlation matrix for the dataset
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Owner</th>
      <th>Num_Years</th>
      <th>Fuel_Type_Diesel</th>
      <th>Fuel_Type_Petrol</th>
      <th>Seller_Type_Individual</th>
      <th>Transmission_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Selling_Price</th>
      <td>1.000000</td>
      <td>0.878983</td>
      <td>0.029187</td>
      <td>-0.088344</td>
      <td>-0.236141</td>
      <td>0.552339</td>
      <td>-0.540571</td>
      <td>-0.550724</td>
      <td>-0.367128</td>
    </tr>
    <tr>
      <th>Present_Price</th>
      <td>0.878983</td>
      <td>1.000000</td>
      <td>0.203647</td>
      <td>0.008057</td>
      <td>0.047584</td>
      <td>0.473306</td>
      <td>-0.465244</td>
      <td>-0.512030</td>
      <td>-0.348715</td>
    </tr>
    <tr>
      <th>Kms_Driven</th>
      <td>0.029187</td>
      <td>0.203647</td>
      <td>1.000000</td>
      <td>0.089216</td>
      <td>0.524342</td>
      <td>0.172515</td>
      <td>-0.172874</td>
      <td>-0.101419</td>
      <td>-0.162510</td>
    </tr>
    <tr>
      <th>Owner</th>
      <td>-0.088344</td>
      <td>0.008057</td>
      <td>0.089216</td>
      <td>1.000000</td>
      <td>0.182104</td>
      <td>-0.053469</td>
      <td>0.055687</td>
      <td>0.124269</td>
      <td>-0.050316</td>
    </tr>
    <tr>
      <th>Num_Years</th>
      <td>-0.236141</td>
      <td>0.047584</td>
      <td>0.524342</td>
      <td>0.182104</td>
      <td>1.000000</td>
      <td>-0.064315</td>
      <td>0.059959</td>
      <td>0.039896</td>
      <td>-0.000394</td>
    </tr>
    <tr>
      <th>Fuel_Type_Diesel</th>
      <td>0.552339</td>
      <td>0.473306</td>
      <td>0.172515</td>
      <td>-0.053469</td>
      <td>-0.064315</td>
      <td>1.000000</td>
      <td>-0.979648</td>
      <td>-0.350467</td>
      <td>-0.098643</td>
    </tr>
    <tr>
      <th>Fuel_Type_Petrol</th>
      <td>-0.540571</td>
      <td>-0.465244</td>
      <td>-0.172874</td>
      <td>0.055687</td>
      <td>0.059959</td>
      <td>-0.979648</td>
      <td>1.000000</td>
      <td>0.358321</td>
      <td>0.091013</td>
    </tr>
    <tr>
      <th>Seller_Type_Individual</th>
      <td>-0.550724</td>
      <td>-0.512030</td>
      <td>-0.101419</td>
      <td>0.124269</td>
      <td>0.039896</td>
      <td>-0.350467</td>
      <td>0.358321</td>
      <td>1.000000</td>
      <td>0.063240</td>
    </tr>
    <tr>
      <th>Transmission_Manual</th>
      <td>-0.367128</td>
      <td>-0.348715</td>
      <td>-0.162510</td>
      <td>-0.050316</td>
      <td>-0.000394</td>
      <td>-0.098643</td>
      <td>0.091013</td>
      <td>0.063240</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating a Resale Value Percentage feature for understanding data
df['Resale_Percentage'] = round(df['Selling_Price'] / df['Present_Price'] * 100, 2)
```


```python
# Plotting the pairplot for the dataset
sns.pairplot(df[['Selling_Price', 'Present_Price', 'Kms_Driven', 'Num_Years', 'Resale_Percentage']])
```




    <seaborn.axisgrid.PairGrid at 0x28924ed2640>




    
![png](outputs/output_13_1.png)
    



```python
# Plotting the correlation heatmap
plt.figure(figsize = (20, 20))
sns.heatmap(df.corr(), annot = True, cmap = 'RdBu')
```




    <AxesSubplot:>




    
![png](outputs/output_14_1.png)
    



```python
# Dropping the 'Resale_Percentage' feature
df.drop(['Resale_Percentage'], axis = 1, inplace = True)
```


```python
# Extracting dependent and independent features
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Owner</th>
      <th>Num_Years</th>
      <th>Fuel_Type_Diesel</th>
      <th>Fuel_Type_Petrol</th>
      <th>Seller_Type_Individual</th>
      <th>Transmission_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.59</td>
      <td>27000</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.54</td>
      <td>43000</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.85</td>
      <td>6900</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.15</td>
      <td>5200</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.87</td>
      <td>42450</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




    0    3.35
    1    4.75
    2    7.25
    3    2.85
    4    4.60
    Name: Selling_Price, dtype: float64




```python
# Getting feature importances
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)
```

    [0.401019   0.04184067 0.00048337 0.07637049 0.21458786 0.01396357
     0.1189945  0.13274053]
    


```python
# Plotting barplot feature importances
feature_imp = pd.Series(model.feature_importances_, index = X.columns)
feature_imp.plot(kind = 'barh')
plt.show()
```


    
![png](outputs/output_20_0.png)
    



```python
# Dropping 'Owner' feature as its importance is very low
df.drop(['Owner'], axis = 1, inplace = True)
```


```python
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
```

    [0.37338901 0.03971805 0.07811791 0.22399295 0.01095715 0.13752305
     0.13630188]
    


    
![png](outputs/output_22_1.png)
    



```python
# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)
```

#### Linear Regression


```python
# Creating a linear regression model
linear_regressor = LinearRegression(n_jobs=-1)
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)
```


```python
# Computing some performance metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))
```

    Mean Absolute Error: 1.2346931037304398
    Mean Squared Error: 4.196277303132508
    Root Mean Squared Error: 2.04848170680934
    R2 Score: 0.8467812557718505
    


```python
# Plotting histogram of difference between y_pred and y_test
sns.displot(y_test - y_pred, kind = 'kde')
plt.xlabel('Selling Price (y_test - y_pred)')
plt.show()
```


    
![png](outputs/output_27_0.png)
    



```python
# Plotting scatterplot between y_pred and y_test
plt.scatter(y_test, y_pred)
plt.title('True Value vs Predicted Value')
plt.ylabel('y_test')
plt.xlabel('y_pred')
plt.show()
```


    
![png](outputs/output_28_0.png)
    


#### Support Vector Regression


```python
# Creating a hyperparamter value grid for hyperparamter tuning
random_grid = {
    'C' : [0.1, 1, 10],
    'epsilon' : [0.01, 0.1, 1]
}
```


```python
# Creating a RandomSearchCV model
support_vector_regressor = RandomizedSearchCV(estimator=SVR(), param_distributions=random_grid,
                                              scoring='neg_mean_squared_error', n_iter=10, cv=5, 
                                              verbose=1, n_jobs=-1)
support_vector_regressor.fit(X_train, y_train)
```

    C:\Users\RISHABH\anaconda3\lib\site-packages\sklearn\model_selection\_search.py:285: UserWarning: The total space of parameters 9 is smaller than n_iter=10. Running 9 iterations. For exhaustive searches, use GridSearchCV.
      warnings.warn(
    

    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    




    RandomizedSearchCV(cv=5, estimator=SVR(), n_jobs=-1,
                       param_distributions={'C': [0.1, 1, 10],
                                            'epsilon': [0.01, 0.1, 1]},
                       scoring='neg_mean_squared_error', verbose=1)




```python
# Printing best score and best hyperparameter values
print('Best Score:', support_vector_regressor.best_score_)
print('Best Hyperparamters:', support_vector_regressor.best_params_)
```

    Best Score: -18.488439560490036
    Best Hyperparamters: {'epsilon': 1, 'C': 10}
    


```python
# Creating final model with best hyperparameter values
support_vector_regressor = SVR(C=10, epsilon=1)
support_vector_regressor.fit(X_train, y_train)
y_pred = support_vector_regressor.predict(X_test)
```


```python
# Computing some performance metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))
```

    Mean Absolute Error: 3.473097408420735
    Mean Squared Error: 30.072223069123147
    Root Mean Squared Error: 5.483814645766499
    R2 Score: -0.09802759015955798
    


```python
# Plotting histogram of difference between y_pred and y_test
sns.displot(y_test - y_pred, kind = 'kde')
plt.xlabel('Selling Price (y_test - y_pred)')
plt.show()
```


    
![png](outputs/output_35_0.png)
    



```python
# Plotting scatterplot between y_pred and y_test
plt.scatter(y_test, y_pred)
plt.title('True Value vs Predicted Value')
plt.ylabel('y_test')
plt.xlabel('y_pred')
plt.show()
```


    
![png](outputs/output_36_0.png)
    


#### Random Forest Regression


```python
# Creating a default (simple) Random Forest Regressor model
random_forest_regressor = RandomForestRegressor(n_jobs=-1)
random_forest_regressor.fit(X_train, y_train)
y_pred = random_forest_regressor.predict(X_test)

# Performance Metrics
print('R2 score: ', r2_score(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
```

    R2 score:  0.7078442261091236
    Mean Squared Error:  8.00141424688796
    


```python
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
```


```python
# Creating a RandomSearchCV model
random_forest_regressor = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
                                             scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=1, n_jobs=-1)
random_forest_regressor.fit(X_train, y_train)
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    




    RandomizedSearchCV(cv=5, estimator=RandomForestRegressor(), n_jobs=-1,
                       param_distributions={'max_depth': [5, 10, 15, 20, 25, 30],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 5, 10],
                                            'min_samples_split': [2, 5, 10, 15,
                                                                  100],
                                            'n_estimators': [100, 200, 300, 400,
                                                             500, 600, 700, 800,
                                                             900, 1000, 1100,
                                                             1200]},
                       scoring='neg_mean_squared_error', verbose=1)




```python
# Printing the best parametets and best score
print('Best Parameter:', random_forest_regressor.best_params_)
print('Best Score:', random_forest_regressor.best_score_)
```

    Best Parameter: {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 20}
    Best Score: -1.7018199496759348
    


```python
# Creating a final random Forest Regression model with best hyperparameter values
random_forest_regressor = RandomForestRegressor(n_estimators=1000, min_samples_split=2, min_samples_leaf=1,
                                                max_features='auto', max_depth=20, n_jobs=-1)
random_forest_regressor.fit(X_train, y_train)
y_pred = random_forest_regressor.predict(X_test)
```


```python
# Computing some performance metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))
```

    Mean Absolute Error: 1.278691618257259
    Mean Squared Error: 7.574130307710376
    Root Mean Squared Error: 2.752113789019338
    R2 Score: 0.7234456518158505
    


```python
# Plotting histogram of difference between y_pred and y_test
sns.displot(y_test - y_pred, kind = 'kde')
plt.xlabel('Selling Price (y_test - y_pred)')
plt.show()
```


    
![png](outputs/output_44_0.png)
    



```python
# Plotting scatterplot between y_pred and y_test
plt.scatter(y_test, y_pred)
plt.title('True Value vs Predicted Value')
plt.ylabel('y_test')
plt.xlabel('y_pred')
plt.show()
```


    
![png](outputs/output_45_0.png)
    



```python
# Exporting the models to a pickle file

# Open file in desired location/directory
file = open('models.pkl', 'wb') 
# Dump information to that file
pickle.dump([linear_regressor, support_vector_regressor, random_forest_regressor], file)
# Close the file
file.close()
```
