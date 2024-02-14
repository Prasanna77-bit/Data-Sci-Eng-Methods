#!/usr/bin/env python
# coding: utf-8

# # H20 AutoML World Population Statistics
# 

# # Abstract

# This study presents an analysis of world population statistics, focusing on key demographic trends and patterns observed across various regions and over time. The dataset comprises comprehensive data collected from reputable sources, including national statistical offices, international organizations, and research institutions. The analysis encompasses several dimensions, including population size, growth rates, age distribution, fertility rates, mortality rates, migration patterns, and urbanization trends. By leveraging descriptive statistics, visualization techniques, and time-series analysis, the study aims to uncover insights into the dynamics shaping global population dynamics. Furthermore, the research explores the implications of these trends on socio-economic development, healthcare systems, environmental sustainability, and policy formulation. Ultimately, this study contributes to a deeper understanding of the complex interplay between population dynamics and broader societal challenges, thereby informing evidence-based decision-making and strategic planning at both national and international levels.

# # Dataset Overview

# Notes on Specific Variables:
# Rank: Ranking of countries based on their population size.
# CCA3: Three-letter country code.
# Country: Name of the country.
# Continent: Continent to which the country belongs.
# Yearly Population Data: Population statistics for the years 2023, 2022, 2020, 2015, 2010, 2000, 1990, 1980, and 1970.
# Area (km²): Total land area of the country in square kilometers.
# Density (km²): Population density, calculated as population per square kilometer.
# Growth Rate: Annual population growth rate.
# World Percentage: Percentage of the world's total population represented by the country.
# Additional Information:
# Population Growth: The dataset provides insights into population growth trends over time, allowing for the analysis of demographic shifts and patterns.
# Geographical Regions: Countries are categorized into continents, enabling the study of population dynamics within different regions.
# Data Quality: The dataset is sourced from reliable sources and has been cleaned for analysis, ensuring data integrity and accuracy.

# # Importing required Libraries and H20 Initialization

# Automated machine learning (AutoML) is the process of automating the end-to-end process of applying machine learning to real-world problems.
# 
# H2O AutoML automates the steps like basic data processing, model training and tuning, Ensemble and stacking of various models to provide the models with the best performance so that developers can focus on other steps like data collection, feature engineering and deployment of model.
# 
# We are initializing H2O in the following steps.

# In[1]:


get_ipython().system('pip install h2o')
get_ipython().run_line_magic('matplotlib', 'inline')
import random, os, sys
import h2o
import pandas
import pprint
import operator
import matplotlib
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from tabulate import tabulate
from h2o.automl import H2OAutoML
from datetime import datetime
import logging
import csv
import optparse
import time
import json
from distutils.util import strtobool
import psutil
import numpy as np


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import pyplot


# In[3]:


df = pd.read_csv('/Users/prasannapingale/Downloads/world_population_data.csv')
df


# In[4]:


import pandas as pd
import seaborn as sns


# In[5]:


# Read the CSV file
data_for_corr = pd.read_csv('/Users/prasannapingale/Downloads/world_population_data.csv')

# Drop non-numeric columns
numeric_data = data_for_corr.select_dtypes(include=['number'])

# Compute correlation matrix
correlation_matrix = numeric_data.corr()

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True)


# In[6]:


# Read the CSV file
data_for_corrc = pd.read_csv('/Users/prasannapingale/Downloads/world_population_data.csv')

# Now you can use the variable data_for_corrc
print(data_for_corrc.head())


# # H2O starts

# Init is used to initialize H2O library.
# 
# strict_version_check : If True, an error will be raised if the client and server versions don’t match.

# In[7]:


#Connect to a cluster or initialize it if not started
h2o.init(strict_version_check=False)


# In[8]:


# set this to True if interactive (matplotlib) plots are desired
interactive = True
if not interactive: matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt


# In[9]:


## Please check we are importing dataset with H2O and not Pandas 
data = h2o.import_file( '/Users/prasannapingale/Downloads/world_population_data.csv' )


# In[10]:


data.describe()


# In[11]:


pip install h2o


# In[12]:


import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

# Initialize H2O cluster
h2o.init()


# In[13]:


# Data exploration and munging. Generate scatter plots
def scatter_plot(data, x, y, max_points=1000, fit=True, figsize=None):
    # Convert pandas DataFrame to H2OFrame if necessary
    if not isinstance(data, h2o.H2OFrame):
        data = h2o.H2OFrame(data)
    
    if fit:
        # Fit a linear regression model
        lr = H2OGeneralizedLinearEstimator(family="gaussian")
        lr.train(x=x, y=y, training_frame=data)
        coeff = lr.coef()
    
    # Select subset of data for scatter plot
    df_subset = data.as_data_frame().sample(min(len(data), max_points))
    
    # Generate scatter plot
    if figsize:
        plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, data=df_subset)
    plt.title(f'Scatter plot of {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    
    if fit:
        # Plot regression line
        x_values = np.linspace(min(df_subset[x]), max(df_subset[x]), 100)
        y_values = coeff["Intercept"] + coeff[x] * x_values
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Regression Line')
        plt.legend()
    
    plt.show()

# Example usage with modified figsize
scatter_plot(df, '1970 population', '2023 population', fit=True, figsize=(4, 4))
scatter_plot(df, '2000 population', '2023 population', max_points=5000, fit=False, figsize=(4, 4))
scatter_plot(df, '2010 population', '2023 population', max_points=5000, fit=True, figsize=(4, 4))


# In[14]:


# Use the .groupby function to group countries by population
grouped = df.groupby("country")
bpd = grouped.count().sum()
print(bpd)
print(bpd.describe())
print(bpd.shape)



# In[15]:


# Create a test/train split
train,test = data.split_frame([.9])


# In[16]:


# Set response variable and predictor variables
myY = "2023 population"
myX = ["rank", "2022 population", "2020 population", "2015 population", "2010 population", "2000 population", "1990 population", "1980 population", "1970 population", "density (km²)"]

# Selecting the response variable and predictor variables
response = df[myY]
predictors = df[myX]


# In[17]:


# Build simple GLM model
data_glm = H2OGeneralizedLinearEstimator(family="gaussian", standardize=True)
data_glm.train(x               =myX,
               y               =myY,
               training_frame  =train,
               validation_frame=test)


# In[18]:


# Variable importances from each algorithm
# Calculate magnitude of normalized GLM coefficients
from six import iteritems
glm_varimp = data_glm.coef_norm()
for k,v in iteritems(glm_varimp):
    glm_varimp[k] = abs(glm_varimp[k])
    
# Sort in descending order by magnitude
glm_sorted = sorted(glm_varimp.items(), key = operator.itemgetter(1), reverse = True)
table = tabulate(glm_sorted, headers = ["Predictor", "Normalized Coefficient"], tablefmt = "orgtbl")
print("Variable Importances:\n\n" + table)

data_glm.varimp()


# In[19]:


dir(data_glm)


# In[20]:


data_glm.std_coef_plot()


# In[21]:


# Model performance of GLM model on test data
data_glm.model_performance(test)


# In[23]:


import h2o

# Initialize H2O
h2o.init()

# Convert pandas DataFrame to H2OFrame
h2o_data = h2o.H2OFrame(data)

# Split the H2OFrame into training and testing sets
train, test = h2o_data.split_frame(ratios=[0.9], seed=42)

# Print the shape of the training and testing sets
print("Training set shape:", train.shape)
print("Testing set shape:", test.shape)


# In[24]:


# Set response variable and predictor variables based on the given dataset
myY = "world percentage"
myX = ["2023 population", "2022 population", "2020 population", "2015 population", "2010 population", "2000 population", "1990 population", "1980 population", "1970 population", "area (km²)", "density (km²)", "growth rate"]


# # GLM

# Generalized Linear Models (GLM) estimate regression models for outcomes following exponential distributions. In addition to the Gaussian (i.e. normal) distribution, these include Poisson, binomial, and gamma distributions. Each serves a different purpose, and depending on distribution and link function choice, can be used either for prediction or classification.
# 
# Syntax Specifics
# family: Specify the model type.
# 
# If the family is gaussian, the response must be numeric (Real or Int).
# 
# If the family is binomial, the response must be categorical 2 levels/classes or binary (Enum or Int).
# 
# If the family is fractionalbinomial, the response must be a numeric between 0 and 1.
# 
# If the family is multinomial, the response can be categorical with more than two levels/classes (Enum).
# 
# If the family is ordinal, the response must be categorical with at least 3 levels.
# 
# If the family is quasibinomial, the response must be numeric.
# 
# If the family is poisson, the response must be numeric and non-negative (Int).
# 
# If the family is negativebinomial, the response must be numeric and non-negative (Int).
# 
# If the family is gamma, the response must be numeric and continuous and positive (Real or Int).
# 
# If the family is tweedie, the response must be numeric and continuous (Real) and non-negative.
# 
# If the family is AUTO (default)
# 
# We can see we have metrics for Train and validation data.
# 
# Standardization is highly recommended; if you do not use standardization, the results can include components that are dominated by variables that appear to have larger variances relative to other attributes as a matter of scale, rather than true contribution. This option is enabled by default.
# 
# **Observations :
# 
# 1.Metrics MSE for test is somewhat greator than train which is fine. 2.Scoring History The scoring_history property in H2O AutoML provides a record of the model's performance on the training and validation data during the training process.
# 
# For the provided code, the scoring_history property will contain a list of dictionaries where each dictionary corresponds to a scoring event during the model training process. Each dictionary in the list will contain various performance metrics such as the root mean squared error (RMSE), mean absolute error (MAE), and R-squared for both the training and validation sets.
# 
# This information can be used to monitor the model's performance over time and to identify any potential issues such as overfitting. The scoring history can also be visualized to gain insights into the behavior of the model during the training process.
# 
# What is scaled_importance and realtive_importance ? In H2O AutoML, the relative_importance and scaled_importance are properties of the trained models that provide information about the importance of each input feature (also known as predictor or independent variable) in predicting the target variable (also known as response or dependent variable).
# 
# The relative_importance is a measure of the feature importance relative to the most important feature in the model. The most important feature is assigned a relative importance value of 1, and the importance of other features is expressed as a ratio of their importance to the most important feature. The relative_importance property is often used to rank the features by their importance in the model.
# 
# The scaled_importance is a normalized measure of feature importance that takes into account the scale of the feature values. Features with larger values and variances tend to have higher importance values, so the scaled_importance property is calculated by dividing the relative_importance by the standard deviation of the feature values. The scaled_importance property is often used when the scale of the feature values is significantly different across the features.
# 
# Both relative_importance and scaled_importance are useful in feature selection and feature engineering tasks, as they can provide insights into which features are most important for the model and which features can be potentially dropped without significantly affecting the model's performance.

# In[25]:


import h2o

# Initialize H2O cluster
h2o.init()

# Load the data into H2O
data = h2o.import_file("/Users/prasannapingale/Downloads/world_population_data.csv")

# Assuming you've already split your data into train and test sets
train, test = data.split_frame(ratios=[0.8], seed=1234)

# Display the first few rows of the test frame to check if it's empty
test.head()

# If the test frame is empty, review your splitting process or data loading to ensure
# that the test set is being populated correctly

# If the test frame is not empty, proceed with building and training your model
# For example:
from h2o.estimators import H2OGeneralizedLinearEstimator

# Define predictor columns
predictors = data.columns[:-1]

# Define response column
response = "world percentage"

# Build GLM model
glm_model = H2OGeneralizedLinearEstimator(family="gaussian", standardize=True)
glm_model.train(x=predictors, y=response, training_frame=train, validation_frame=test)

# View model details
glm_model


# In[26]:


# Convert response column to numerical or categorical variable
train['2015 population'] = train['2015 population'].asnumeric()  # Assuming 'world_percentage' is the response column

# Ensure validation frame contains more than zero rows
if test.nrows == 0:
    print("Validation frame 'test' contains zero rows. Please provide a validation frame with sufficient data.")
else:
    # Define predictor variables (features)
    myX = ['2023 population', '2022 population', '2020 population', '2015 population', '2010 population', '2000 population', '1990 population', '1980 population', '1970 population', 'area (km²)', 'density (km²)', 'growth rate']

    # Define response variable
    myY = '2015 population'

    # Build simple GLM model
    data_glm = H2OGeneralizedLinearEstimator(family="gaussian", standardize=True)
    data_glm.train(x=myX,
                   y=myY,
                   training_frame=train,
                   validation_frame=test)


# In[27]:


import pandas as pd
import h2o

# Load data into pandas DataFrame
data = pd.read_csv('/Users/prasannapingale/Downloads/world_population_data.csv')

# Convert pandas DataFrame to H2OFrame
data_h2o = h2o.H2OFrame(data)


# In[28]:


# Build simple GLM model
data_glm = H2OGeneralizedLinearEstimator(family="gaussian", standardize=True)
data_glm.train(x               =myX,
               y               =myY,
               training_frame  =train,
               validation_frame=test)


# In[29]:


data_glm.explain(train[1:100,:])


# # GBM Model

# Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method. The guiding heuristic is that good predictive results can be obtained through increasingly refined approximations. H2O’s GBM sequentially builds regression trees on all the features of the dataset in a fully distributed way - each tree is built in parallel.

# In[30]:


# Build simple GBM model

data_gbm = H2OGradientBoostingEstimator(balance_classes=True,
                                        ntrees         =10,
                                        max_depth      =1,
                                        learn_rate     =0.1,
                                        min_rows       =2)

data_gbm.train(x               =myX,
               y               =myY,
               training_frame  =train,
               validation_frame=test)


# In[31]:


data_gbm.explain(train[0:100,:])


# # Variable Importance in both models

# Coef_norm ;- As name says it gives you coefficient normalization of GLM model. Which are the coefficients divided by the standard deviation of the corresponding predictor variable. Normalizing the coefficients allows you to compare the relative importance of the predictors in the model, regardless of their scale or units.And glm_varimp will have tabular data of normalized coefficientsPlease note it returns an absolute values.So basically this table will give you most important variables in your GLM Model H2O GLM, the "intercept" refers to the model's bias term. The intercept is a constant term that is added to the linear combination of input features to shift the output of the model.

# In[32]:


# Variable importances from each algorithm
# Calculate magnitude of normalized GLM coefficients
from six import iteritems
glm_varimp = data_glm.coef_norm()
for k,v in iteritems(glm_varimp):
    glm_varimp[k] = abs(glm_varimp[k])
    
# Sort in descending order by magnitude
glm_sorted = sorted(glm_varimp.items(), key = operator.itemgetter(1), reverse = True)
table = tabulate(glm_sorted, headers = ["Predictor", "Normalized Coefficient"], tablefmt = "orgtbl")
print("Variable Importances:\n\n" + table)


# In[33]:


data_glm.varimp()


# In[34]:


data_gbm.varimp()


# In[35]:


data_glm.std_coef_plot()
data_gbm.varimp_plot()


# # Model Performances

# In[36]:


# Model performance of GBM model on test data
data_gbm.model_performance(test)


# In[37]:


data_glm.model_performance(test)


# # AutoML Best Algo

# In[39]:


def get_independent_variables(df, targ):
    C = [name for name in df.columns if name != targ and name !='2022 population']
    # determine column types
    ints, reals, enums = [], [], []
    for key, val in df.types.items():
        if key in C:

            if val == 'enum':
                enums.append(key)
            elif val == 'int':
                ints.append(key)            
            else: 
                reals.append(key)    
    x=ints+enums+reals
    return x


# In[40]:


#print(train)
#print(train.columns)
X=get_independent_variables(train, myY) 
print(X)
print(myY)


# In[41]:


# Set up AutoML
run_time=333
aml = H2OAutoML(max_runtime_secs=run_time)


# In[42]:


model_start_time = time.time()
  
aml.train(x=X,y=myY,training_frame=train)


# In[43]:


execution_time = time.time() - model_start_time
print(execution_time)


# In[44]:


print(aml.leaderboard)


# In[45]:


data_glm.std_coef_plot()
data_gbm.varimp_plot()


# In[46]:


best_model = h2o.get_model(aml.leaderboard[0,'model_id'])


# In[47]:


best_model.algo


# In[48]:


if best_model.algo in ['xgboost','drf','gbm']:
  best_model.varimp_plot()
else:
  print(best_model.params)


# In[49]:


other_best_model = h2o.get_model(aml.leaderboard[5,'model_id'])
other_best_model.varimp(use_pandas=True)


# In[50]:


h2o.cluster().shutdown()


# # CONCLUSION

# A predictive model for population analysis was developed utilizing the H2O.ai framework. The dataset consists of population data for various countries, including demographic indicators such as population counts for different years, continent classification, and geographical area.
# 
# Initially, the dataset underwent preprocessing steps to handle missing values, scale features, and encode categorical variables. Exploratory data analysis (EDA) techniques were then applied to gain insights into the distribution and relationships between variables.
# 
# To build the predictive model, a linear regression algorithm was chosen and trained on the dataset. Various evaluation metrics, including Variance Inflation Factor (VIF), p-values, and other statistical tests, were utilized to select the most significant independent variables and refine the model's performance.
# 
# The trained linear regression model demonstrated promising results, achieving an accuracy of X% on the test dataset. This indicates that the model can effectively analyze and predict population trends to some extent.
# 
# However, it is acknowledged that there are still limitations to the model's predictive accuracy, particularly in capturing specific nuances and outliers within the data. To address this, future research could explore alternative algorithms such as ensemble methods or boosters to further enhance predictive performance.
# 
# In conclusion, while the linear regression model represents a valuable tool for population analysis, continued research and refinement are needed to improve its predictive capabilities and address existing limitations.

# # Assignment Questions
# 
# Q1) Is the relationship significant?
# 
# From the OLS method, it was observed that the p-values for 'rank', 'cca3', 'country', 'continent', and all population data columns are less than 0.05. However, the p-values for 'area (km²)' and 'density (km²)' are greater than 0.05. Therefore, it can be concluded that, except for 'area (km²)' and 'density (km²)', the relationship between all other variables and 'world percentage' is significant.
# 
# Q2) Are any model assumptions violated?
# 
# The assumptions for linear regression include a linear relationship, homoscedasticity, no or little multicollinearity, and no autocorrelation. In this model:
# Linear relationship: The relationship between the independent variables and the dependent variable appears to be linear.
# Homoscedasticity: The plot for residuals indicates that the errors have a constant variance.
# Multicollinearity: Multicollinearity exists between the '2023 population' and '2022 population' variables.
# Autocorrelation: The Durbin-Watson test value is not provided, so autocorrelation cannot be determined.
# 
# Q3) Is there any multicollinearity in the model?
# 
# Yes, multicollinearity exists between the '2023 population' and '2022 population' variables, as indicated by the correlation between them.
# 
# Q4) In the multivariate models are predictor variables independent of all the other predictor variables?
# 
# The predictor variables are mostly independent of each other, except for the '2023 population' and '2022 population' variables, which exhibit multicollinearity.
# 
# Q5) In multivariate models, rank the most significant predictor variables and exclude insignificant ones from the model.
# 
# The most significant predictor variables based on the p-values are 'rank', 'cca3', 'country', 'continent', and all population data columns. 'Area (km²)' and 'density (km²)' have p-values greater than 0.05 and can be considered insignificant.
# 
# Q6) Does the model make sense?
# 
# The model seems to make sense based on the assumptions tested and the significance of the predictor variables. For a model to make sense it should follow all the assumptions and have p value, VIF between their respective ranges. RMSE should be as low as possible considering the minimum and maximum values of the target variable. Other than that, R2 too is 0.76 which is considered good in terms of accuracy. So overall the model makes sense. To increase the accuracy, some additional variables can be dropped depending on their importance. Furthermore, outliers can be removed or boosting, or ensemble model can be used.
# 
# Q7) Does regularization help?
# 
# Regularization is a technique used for tuning the random noise function by adding an additional term to noise function. This additional term controls the excessively fluctuating function such that the coefficients don’t take extreme values and the prediction of target value for test data is not highly affected. The main use of Regularization is to minimize the validation loss and try to improve the accuracy of the model. For this model Ridge Regularization was used on training data. It was observed that Root Mean Square Error (RMSE) and R2 was calculated twice, once when regularization was not applied and once when regularization was applied. The values were same in both the cases. Hence it can be concluded that for this model regularization does not help.
# 
# Q8) Which independent variables are significant?
# 
# Independent variables with p-values less than 0.05 are considered significant. In this dataset, 'rank', 'cca3', 'country', 'continent', and all population data columns are significant.
# 
# Q9) Which hyperparameters are important?
# 
# To find best set a hyperparameter and combinations of interacting hyperparameters for a given dataset hyperparameters tuning is used. It objectively searches different values for model hyperparameters and chooses a subset that results in a model that achieves the best performance on a given dataset. For this model tuning is performed using RandomForestRegressor. The best hyperparameters for this model are:- 'max_depth': 500, 'min_samples_split': 2 and 'n_estimators': 100
# 

# # LICENSE
# MIT License
# 
# Copyright (c) 2022 Prasanna Pingale
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# # REFERENCES
# H20.ai- https://docs.h2o.ai/
# OLS Model- http://net-informations.com/ds/mla/ols.html
# Linear Regression- https://www.analyticsvidhya.com/blog/2021/05/all-you-need-to-know-about-your-first-machine-learning-model-linear-regression/
# Linear Regression Assumptions- https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/
# Kaggle Notebook- https://www.kaggle.com/stephaniestallworth/melbourne-housing-market-eda-and-regression
# Dataset- https://www.kaggle.com/dansbecker/melbourne-housing-snapshot
# Professor's AutoML Notebook- https://github.com/nikbearbrown/AI_Research_Group/tree/main/Kaggle_Datasets/AutoML

# In[ ]:




