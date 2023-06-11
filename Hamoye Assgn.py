#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


#Loading our data
data = pd.read_csv("https://drive.google.com/file/d/1Eru_UHVc3WLHVveC9Q8K9QUxlzYeHt18/view.csv")
data.head()


# In[3]:


#Describing our data
data.describe().T


# In[4]:


#Checking on data types
data.dtypes


# In[5]:


#Checking for null values
data.isnull().sum()


# Our dataset has no missing values.

# In[6]:


#Checking for duplicated in our dataset
data.duplicated().sum()


# There is no duplicate in our dataset.

# In[7]:


#Identifying our dependent and independent variables
X = data[['T2']]  # Independent variable (living room temperature)
y = data['T6']    # Dependent variable (outside temperature)


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# Remove the specified columns
columns_to_remove = ["date", "lights"]
data = data.drop(columns=columns_to_remove)

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler()
normalised_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
X = normalised_data.drop(columns=['Appliances'])
y = normalised_data['Appliances']


# In[9]:


#Now, we split our dataset into the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[10]:


#Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

#Fit the model to the training set
linear_model.fit(X_train, y_train)

#obtaining predictions
predicted_values = linear_model.predict(X_test)


# In[11]:


#Mean Absolute Error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 2)


# In[12]:


#Finding the R2
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# In[14]:


#The residual sum of squares
rss = np.sum((y_test - predicted_values) ** 2)
round(rss, 2)


# In[16]:


#The Root Mean Squared Error
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[23]:


from sklearn.linear_model import Ridge
ridge_reg =  Ridge(alpha = 0.5)
ridge_reg.fit(X_train, y_train)


# In[25]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)


# In[26]:


#Comparing the effects of regularisation
def get_weights_data(model, feat, col_name):#This function returns the weight of every feature
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_data = pd.DataFrame(weights).reset_index()
    weights_data.columns = ['features', col_name]
    weights_data[col_name].round(3)
    return weights_data


# In[27]:


linear_model_weights = get_weights_data(linear_model, X_train, 'Linear_Model_Weight')
ridge_weights_data = get_weights_data(ridge_reg, X_train, 'Ridge_Weight')
lasso_weights_data = get_weights_data(lasso_reg, X_train, 'Lasso_weight')
final_weights = pd.merge(linear_model_weights, ridge_weights_data, on='features')
final_weights = pd.merge(final_weights, lasso_weights_data, on='features')


# In[28]:


print(final_weights)


# From the question, RH_2 and RH_1 has the lowest and the highest weights respectively, in the linear model.

# In[30]:


#Train a ridge regression model with an alpha value of 0.4. 
#Is there any change to the root mean squared error (RMSE) when evaluated on the test set?
ridge_reg =  Ridge(alpha = 0.4)
ridge_reg.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# There is no change.
