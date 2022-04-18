# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from collections import Counter 
import os
import dataframe_image as dfi

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

sb.set(font_scale = 1.9)




#Functions for the this project

#data cleaning functions
def null_cleaning(data,columns):
    df =data
    for column in columns:
         df = df[df[column].notna()]
    return df

def value_cleaning(data, columns, value):
    df = data
    for column in columns:
        df = df[df[column] > value];
    return df

def outlier_cleaning(data, columns):
    df = data
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) |(df[column] > (Q3 + 1.5 * IQR)))]
    return df
        

#function for plotting top 10 most popular data
def plot_most_popular(data, region, base):
    publisher = data.groupby(base)[region].sum().reset_index()
    top_10 = publisher.nlargest(10,region)
    f = plt.figure(figsize= (50,10))
    #total sales
    sb.barplot(data=top_10, x=base, y=region, palette="dark")
    game_sales_scatter = data.loc[data[base].isin(top_10[base])]
    game_sales_scatter = game_sales_scatter[game_sales_scatter[region] > 0]
    f = plt.figure(figsize= (50,10))
    sb.boxplot(data=game_sales_scatter, x=base, y=region, palette="deep")

#function for top 5 games in each region
def top5_gamesByRegion(data, region):
    top_5 = data.nlargest(5, region)
    f = plt.figure(figsize= (30, 10))
    sb.barplot(data=top_5, x="Name",y=region, palette="dark")

#function for top 5 genres given region
def top5_genreByRegion(data, region):
    genre = data.groupby("Genre")[region].sum().reset_index()
    top_5 = genre.nlargest(5,region)
    f = plt.figure(figsize= (20,10))
    #total sales
    sb.barplot(data=top_5, x="Genre", y=region, palette="dark")
    game_sales_scatter = data.loc[data["Genre"].isin(top_5['Genre'])]
    print(game_sales_scatter.shape)
    game_sales_scatter = game_sales_scatter[game_sales_scatter[region] > 0]
    f = plt.figure(figsize= (20,10))
    sb.boxplot(data=game_sales_scatter, x="Genre", y=region, palette="deep")

#function to plot userscores given region
def userscore_region(data, region):
    userscore = pd.DataFrame(data["User_Score"])
    regionSales = pd.DataFrame(data[region])
    userscore_train, userscore_test, regionSales_train, regionSales_test = train_test_split(userscore, regionSales, test_size=0.2, random_state=42)
    trainDF = pd.concat([userscore_train, regionSales_train], axis = 1).reindex(userscore_train.index)
    sb.jointplot(data = trainDF, x = "User_Score", y = region, height = 12)
#function to plot criticscore given region
def cscore_region(data, region):
    cscore = pd.DataFrame(data["Critic_Score"])
    regionSales = pd.DataFrame(data[region])
    cscore_train, cscore_test, regionSales_train, regionSales_test = train_test_split(cscore, regionSales, test_size=0.2, random_state=42)
    trainDF = pd.concat([cscore_train, regionSales_train], axis = 1).reindex(cscore_train.index)
    sb.jointplot(data = trainDF, x = "Critic_Score", y = region, height = 12)
    
#Function for correlation heatmap of the data given region
def correlation(data, predictor, region):
    pred = pd.DataFrame(data[predictor])
    sales = pd.DataFrame(data[region])
    pred_train, pred_test, sales_train, sales_test = train_test_split(pred, sales, test_size=0.2, random_state=42)
    trainDF = pd.concat([pred_train, sales_train], axis = 1).reindex(pred_train.index)
    sb.heatmap(trainDF.corr(), vmin = -1, vmax = 1, annot = True, fmt=".2f")
    
#Simple regression function 
def reg(data, sales, predictor):
    pred = pd.DataFrame(data[predictor])
    # Split the Dataset into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(pred,sales, test_size = 0.25, random_state=42)


    # Linear Regression using Train Data
    linreg = LinearRegression()         # create the linear regression object
    linreg.fit(X_train, y_train)        # train the linear regression model

    # Coefficients of the Linear Regression line
    print('Intercept of Regression \t: b = ', linreg.intercept_)
    print('Coefficients of Regression \t: a = ', linreg.coef_)
    print()

    # Predict Total values corresponding to HP
    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)
    
  
    if(type(predictor) == str):
        plt.style.use('default')
        plt.style.use('ggplot')
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(X_train, y_train_pred, color='k', label='Regression model')
        ax.scatter(X_train, y_train, edgecolor='k', facecolor='grey', alpha=0.7, label=predictor)
        ax.set_ylabel('Global_Sales', fontsize=14)
        ax.set_xlabel(str(predictor), fontsize=14)
        ax.legend(facecolor='white', fontsize=11)
        ax.set_title('$R^2= %.2f$' % linreg.score(X_train, y_train), fontsize=18)
        fig.tight_layout()

    # Check the Goodness of Fit (on Train Data)
    print("For predictor "+ str(predictor)+":")
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print()
    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
    print()
    acc = linreg.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_test_pred)    
    return [str(predictor) + "_G,acc, mse]

#simple regression model
def grad(data, sales, predictor):
    X = pd.DataFrame(data[predictor])
    y = sales
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred= reg.predict(X_test)
    acc = reg.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", acc)
    print("Mean Squared Error (MSE) \t:", mse)
    print()
    
    return [str(predictor),acc, mse]

#model function for goodness of fit
def fit_and_eval(data, predictor, sale, model):
    sales = pd.DataFrame(data[sale])
    pred = pd.DataFrame(data[predictor])
    # Split the Dataset into Train and Test
    pred_train, pred_test, sales_train, sales_test = train_test_split(pred,sales, test_size = 0.25, random_state=42)
    
    # Train the model
    model.fit(pred_train, sales_train)
    
    # Make predictions and evalute
    model_pred = model.predict(pred_train)
    
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", model.score(pred_train, sales_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(sales_train, model_pred))


    