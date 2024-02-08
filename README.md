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


The classification problem I would like to propose manipulates two variables, inm_size and inm_distrito, and predicts the price based off of these two factors. Then, the different classifying algorithms would sort the prices into 'bargain' and 'expensive categories. which are sensitive to the average price per square meter in any given district.

My idea was basically to create a column which takes the average price for a property and the average size as features to be addressed; if the price is lower and the square meters are higher than the average for the district then it's considered a bargain. Likewise, if the price per square meter in a property is above the average for the district then it is considered expensive.

-> for this we can nerf all other columns, only include size and price per distrito to create the necessary numerical columns, and encode categorical variables like the category and the distritos.

-> create a new column which states theaverage price for a property in the distrito

-> create a new column which states the average size for a property in the distrito

-> create a new column where you divide the number of square meters into the price (price/size) and that is your price per square meter on average for the distrito.

-> there will only be as many averages as there are distritos, and each data point will have its respective average based on the distrito it is in.

-> we would also have to calculate the price per square meter for that property as well.

-> this means we have at least 4 new columns added to the dataset aside from the distrito one hot encoding.

-> if the price lies is at or less than the average expected price then it is a bargain.

-> if the price of the price per square meter of the property is greater than the average per district then it's expensive.


After I cleaned the dataframe for the linear regression, I created a copy of it which I then used to implement the categorical models. Then, I dropped all columns which were unnecessary aside from those from which I will engineer my new features:

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/a0410083-582b-45f8-bb84-bb47e2c21b1a)

Aside from dummifying 'inm_distrito' like above, I also implemented other blocks of code which created the new features and the thresholds for the categories:
![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/8981e0af-f3ac-42f9-88f7-d9407624d8de)

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/e6dc297b-c9be-4c77-b73c-0677b3335cc2)

After preparing the new dataframe for the different models, it was time to pick the new features and target variables:
![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/40a677be-5e99-49fc-97fb-354cd20f642f)

### Perceptron

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/b7dc3188-181e-4887-ba7c-d15dc83fc074)

A perceptron model is very well-adjusted for binary classification such as this. 

### Logistic Regression

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/4446aa39-8ebf-4521-a362-8cb54498dee8)

The model performs very well! Yay. It was necessary to tweak the maximum number of iterations to learn for the sake of fine tuning the model. If we go over 31 iterations the model overfits and predicts everything perfectly. Overfitting is something we want to avoid because the model may behave very well with a dataset that it is familiar with but not very well with a dataset it hasn't seen before that may follow similar trends to the ones outlined.  

### KNN

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/a8fa641a-a3ee-4e6d-9c30-d60dad55f593)

KNN behaves very well with this classification problem. KNN has learned very well on its training data, but overfitting is a big problem if we want to generalize the model outside of the proposed classification problem. 

Overfitting is something we want to avoid because the model may behave very well with a dataset that it is familiar with but not very well with a dataset it hasn't seen before that may follow similar trends to the ones outlined. 

### LDA
![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/b7d1beaa-c7c2-4a03-9419-fc16b6e016ae)

The great performance for the LDA means that it did a great job of determining whether or not an apartment had good value based on the aformentioned criterion that was possed at the beginning of the classification problem.This makes sense, because sorting the two variables into straightforward categories is easy. 

### QDA

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/4a1b3671-084d-4f65-834a-0bcdfbb309fb)

Therefore, it also makes sense that the performance of the QDA was perhaps too complex for the nature of the problem, and because of this it ended up performing the worst out of all the models. This can be due to the fact that it adds an unnecessary layer of complexity to the proposed problem which is not relevant at the time of sorting. 

### EXTRA MODEL: Decision Trees

![image](https://github.com/solsylph/Models-Madrid-Properties/assets/126614634/6b69a65f-6f90-4938-b6e1-1598a0b03db7)

This decision tree has a very VERY high accuracy. It's concerning, but it is also nice to see which models are best suited for this type of task. 

Overall, I would pick PLA, Logistic Regression, and LDA as the models with the best balance between model flexibility and accuracy. 

## Further improvements

If I were to redo this assignment with a different data set, I would like to test a linear regression with features similar to the ones I picked for an initial build and then determine if factors such as location, size, and index of studied populations also affect the price so significantly. I would also like to look into indecies such as marital index, and number of children per household because these are factors that can influence the affluence of a community/district.

