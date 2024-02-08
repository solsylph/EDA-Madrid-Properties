# EDA-Madrid-Properties

ASSIGNMENT: 

You will find data from several properties in Madrid as well as some historical and sociodemographic data.

1. Analyze this dataset to estimate property prices as a regression problem. Use algorithms seen so far in class and perform exploratory daya analysis (EDA) to determine how you will proceed.
2. Analyze data and redefine the problem to create a classification one.Use the following algorithms and analyze results:
 • Perceptron Learning Algorithm
 • Logistic Regression
 • Generative Models (LDA, QDA)
 • KNN3

## Libraries Used: 

import pandas as pd ##used to handle the dataframes and do operations
import sklearn ##used to train the models, including all modules involve perceptron learning algorithms, logistic regression, knn, LDA, QDA, and decision trees 
import numpy as np ##sift through data and handle it in an easy fashion (especially for KNN)
import matplotlib.pylab as plt ##visualization
import seaborn as sns ##generating plots

## Exploratory Data Analysis
APPROACH:
First, we should identify the data types and features present in the dataset we will have to work with. Then, afterwards, we can begin handling missing data, outliers, etc. 
![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/7430e981-148c-4777-850a-b42989f34e7e)

Based off of this, we can then generate a heat map...

![Untitled](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/d5559bba-20f7-40e2-bc5a-9bfd3ec970f3)

...and then look at the column corresponding to 'inm_price' in order to find which variables have a statistically significant relationship with the target variable. Strong relationships are denoted by having values which approach +1 or -1. 
![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/1eddfbd4-3fd5-4b48-b32d-e14f57ab5c28)

This corresponds to the number of missing variables in the dataset. For the numerical variables we can just fill in the mean wherever there is an empty datapoint (to avoid skewing data and mismanaging results). For columns like 'barrio' and 'distrito', which hold information about a property's location, I coded some functions which could iterate through both columns, create key-value pairs, and update each column's row that had a corresponding distrito.
In this function we can see the logic for this process:

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/b0a78269-67e8-4c6f-9da7-8dc76d5684bf)

In this function we can see how the most commonly occurring barrio in each district is filled into the blank space:
![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/f4487bfa-8f76-4b29-9d73-42e1f373c1e6)

Thanks to the mean imputation for the numerical variables and the imputations described above, I reduced the number of missing values by a factor of 9. 

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/40d134b4-bee6-484f-a2f0-7c75ac3f025c)

Understanding the location of a property was very important to building my model, so I dropped those 185 data points that had no associated barrio or distrito. 

Something important to keep in mind is that you cannot give a machine learning model something like a 'distrito' and allow it to make sense of that feature. You need to manipulate categorical variables, such as labels for districts, in a way that allows the machine to identify the 'uniqueness' through a combination of numbers. This is called encoding, and for my models I chose to perform one-hot encoding, or dumminification. 

Dummification creates a vector of length _n_, where n is the number of unique features which will be extracted from a column with a categorical data type. For example, if I have three different labels in a 'category' column, dummifying the 'category' column would create a vector of length 3, where each unique combination of a 1 and _n-1_ 0s is a numerical 'label' that can be fed into an algorithm. The placement of the '1' in the vector for dummification is typically alphabetical, but it can also just be based on the order of what appears first. 

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/e026141f-a1dd-4b9f-875d-6566c15a0d9d)

Thanks to the dummification of each barrio and distrito, I ended up with a very hefty dataset of size 17,857*193 points with meaningful values (because remember, this is the clean dataset with no missing values). 

After identifying the variables which are significant to how I want to build my model (see the correlation matrix above), I decided to cross-reference the list of potential model features with a dictionary that has the number of outliers per variable in the dataset. In this way, I could perform feature engineering on only the columns which were relevant to building my model instead of wasting my time on transformations which don't contribute to the end result. 

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/1ef52f93-350a-4568-b82f-540e9e127459)

Here is an example of applying a logarithmic transformation to a very skewed data column in order to normalize it and decrease the model's sensitivity to outliers. 

It's easy to do, just one like of code: df.loc[:, 'log_inm_price'] = np.log(df['inm_price'] + 1)

That line of code actually adds the column to the dataframe as well, which is super important when it will come time to train and test the model :)

Finally, after cleaning my dataset and performing EDA, it was time to build the different models used to predict features of the data. 

## Linear Regression

This is how I split my data, removing irrelevant columns and ensuring I used normalized variables to make the model robust:

#we're going to remove 'inm_price' to avoid overfitting the model, along with all of the other features which I don't have a good reason to add.
X = df_merged.drop(columns=['log_inm_price','inm_floor','inm_longitude','inm_latitude','his_quarterly_variation','his_annual_variation','his_monthly_variation','dem_Indice_de_reemplazo_de_la_poblacion_activa','dem_Indice_de_juventud','dem_Indice_de_estructura_de_la_poblacion_activa','dem_Indice_de_dependencia','dem_TasaDeParo','dem_TamanoMedioDelHogar','dem_Proporcion_de_nacidos_fuera_de_Espana','dem_NumViviendas','dem_Densidad_(Habit/Ha)','dem_EdadMedia','dem_PobTotal'])
y = df_merged['log_inm_price']


![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/1c14cefa-d858-4c8f-85b9-d3c1151543b1)

This was the linear regression's overall performance. It's R² value is high enough that it predicts a very significant amount of the data, and fortunately doesn't go so high that I risked overfitting the model. 

## Categorical Problem Remodeling



