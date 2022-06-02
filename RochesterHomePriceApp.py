# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:19:05 2022

@author: T430s
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from PIL import Image
#import sklearn

# loading the saved model for Home Price(random forest model was used)
with open('model_RochesterHomePrice', 'rb') as f1: # rb = read the binary file
    model_price = pickle.load(f1)

# loading the saved model for Tax(random forest model was used)  
with open('model_RochesterHomeTax', 'rb') as f2: 
    model_tax = pickle.load(f2)
   
with open('df_RochesterHomePrice', 'rb') as f3: 
    df = pickle.load(f3)




image = Image.open('home.jpg')
st.image(image)
st.write("""
## Rochester, NY House Price/Tax Prediction 
This app predicts the **Rochester,NY Region House Price by --A. Maharjan**
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

#def user_input_features():
District = st.sidebar.selectbox('Select the District for Price/Tax Prediction', ('gates','greece','chili','penfield','pittsford','webster','brighton','henrietta','East Rochester','Rochester City'))
Bedroom = st.sidebar.selectbox('Bedroom', (1,2,3,4,5,6,7,8,9,10),3)
Bathroom = st.sidebar.selectbox('Bathroom', (1,2,3,4,5,6,7,8,9,10),2)
Area = st.sidebar.number_input('Area',1200,step = 1)
Age = st.sidebar.number_input('How Old is the Home?',40, step = 1)
    
data = {'District': District,
            'Bedroom': Bedroom,
            'Bathroom': Bathroom,
            'Area': Area,
            'Age': Age,
            }
features_price = pd.DataFrame(data, index=[0])

X_new = pd.DataFrame([[District,Bedroom,Bathroom,Area,Age]],columns=['District','Bedroom','Bathroom','Area','Age'])



# Main Panel
# Print specified input parameters
st.header('Specified Input parameters')
st.write(features_price)
st.write('---')

# Apply Model to Make Prediction
prediction_price = model_price.predict(features_price)
features_tax = features_price    
features_tax['Price'] = prediction_price # adding the price column to predict the tax for the property
prediction_tax = model_tax.predict(features_tax)

if st.button('Predict HomePrice'):
    
    st.write('Extimated Home Price is $','%.2f' % prediction_price)
    st.write('Estimated Tax on this home is $','%.2f' % prediction_tax)


#df_input = user_input_features()

# for ANALYTICS
st.sidebar.header('Analytics: Specify District')
district = st.sidebar.selectbox('Select District for Analytics', ('gates','greece','chili','penfield','pittsford','webster','brighton','henrietta','East Rochester','Rochester City'))
if st.sidebar.button('Histogram/BoxPlot of Price'):
    df_district = df[df.District == district]
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(df_district,x='Price',bins = 10)
    plt.xlabel('Price')
    plt.title(district)
    st.pyplot()
    plt.boxplot(df_district['Price'])
    plt.xlabel('Price')
    plt.title(district)
    plt.show()
    st.pyplot()
    st.write(df_district.describe())
    


if st.sidebar.button('Histogram/BoxPlot of Tax'):
    df_district = df[df.District == district]
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(df_district,x='Tax',bins=10)
    plt.xlabel('Tax')
    plt.title(district)
    st.pyplot()
    plt.boxplot(df_district['Tax'])
    plt.xlabel('Tax')
    plt.title(district)
    plt.show()
    st.pyplot()
    st.write(df_district.describe())
    
 
    
# summary sns box plot for all district
st.sidebar.header('Overall Price Distribution')
if st.sidebar.button('Overall Price Distribution by District'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    ax = sns.boxplot(x='District',y='Price',data=df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    st.pyplot()
    
  
if st.sidebar.button('Overall Price Distribution(Histogram)'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(x='Price',data=df)
    st.pyplot()
    st.write(df.Price.describe())
    
    
    
    



st.sidebar.header('Overall Tax Distribution')
if st.sidebar.button('Overall Tax Distribution by District'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    ax = sns.boxplot(x='District',y='Tax',data=df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    st.pyplot()
    
if st.sidebar.button('Overall Tax Distribution(Histogram)'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(x='Tax',data=df)
    st.pyplot()
    st.write(df.Tax.describe())
    
    





# pair plot between the price vs other variables
st.sidebar.header('Correlation')
if st.sidebar.button('Pair/Correlation Plot'):
    
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    #cars_num=cars.select_dtypes(include=['float64','int64','int32'])
    sns.pairplot(df,x_vars = ['Bedroom','Bathroom','Area','Age','Tax'],
             y_vars=['Price'], kind='reg', plot_kws={'line_kws':{'color':'red'}})
    
    #fig = plt.figure(figsize=(12,8)) 
    st.pyplot()



# see which feathres have more contributions
st.sidebar.header('Feature Importance')
if st.sidebar.button('Feature Importance'):
    import shap
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    x = df[['Bedroom','Bathroom','Area','Age']] # does not work with the string and one hot encoded variables district at the momoment for shap
    y = df['Price']
    model.fit(x,y) # fitting with the actual model before piping with all the data points

    explainer = shap.TreeExplainer(model) # does not take the pipe as a model, input the actual model before piping
    shap_values = explainer.shap_values(x)

    plt.title('Feature importance based on SHAP values')
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    shap.summary_plot(shap_values, x)
    st.pyplot(bbox_inches='tight')

    plt.title('Feature importance based on SHAP values (Bar)')
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    shap.summary_plot(shap_values, x, plot_type="bar")
    st.pyplot(bbox_inches='tight')










