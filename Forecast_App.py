############################################################################
###                                                                      ###
##  Time Series Forecast Application on Streamlit using Facebook Prophet  ##
###                                                                      ###
############################################################################

#Designed by Philip Mathew

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fbprophet 
import altair as alt
from scipy.stats import boxcox
from scipy.special import inv_boxcox


# main function
def main():
    
    # title & header
    st.title("Data Forecast Application")
    st.header('Input Time Series Dataset')
    
    # Streamlit file upload - csv only
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    if data_file is not None:  
        df = pd.read_csv(data_file)                 # Reading csv
        columns = list(df.columns.values)            # Extracting column names
        if len(columns) > 2 or len(columns) < 2 :     # Limiting csv's to two columns for forecast                  
            st.write("Error: Please input a csv file with only 2 columns, namely date/time & variable to forecast.")
            st.stop()
        # Extracting respective column names from df    
        elif len(columns) == 2  : 
            val = df.select_dtypes(include=np.number).columns.tolist()   # Extracting variable column 
            date = columns
            date.remove(val[0])           # Extracting date/time column
            df = df[[date[0],val[0]]]      # Reording df columns with date on left
            df_fr = df
            df_fr['ds'] = df_fr[date[0]]       # Column 'ds' for Prophet to use
            df_fr['y'] = df_fr[val[0]]          # Column 'y' for Prophet to use
            st.write('Columns', date[0], '&', val[0], ' were detected in dataset.')
            df = df.rename(columns={date[0]:'index'}).set_index('index')
            st.subheader("Input Dataset:")
            st.dataframe(df[val[0]])
            st.subheader("Summary Statistics:")
            st.write(df.describe())
            ca = alt.Chart(df.reset_index()).encode(x = alt.X('index:T', axis=alt.Axis(title=                       
                                                                  date[0])),y=val[0]).interactive()
            st.subheader("Time Series Plot:")
            st.altair_chart(ca.mark_line(color='firebrick').properties(width=800,height=400))    
            
    # Forecast duration selection
    st.header('Select Forecast Duration:')
    dur = st.number_input('Enter number of periods to forecast in future:',  min_value = 1, step = 1)

    # Forecast Button
    if st.button("Forecast"):
            if data_file is not None:  
                st.spinner()
                with st.spinner(text='Prophet is forecasting:'):
                    
                    # Data-Prep:
                     # Applying Box-Cox Transform to value column and assigning to new column y
                      # Box-Cox Transform evaluates a set of lambda coefficients (λ) and selects the                             
                       # value that achieves the best approximation of normality
                      # Box-Cox will select the λ that maximizes the log-likelihood function 
                    df_fr['y'], lam = boxcox(df_fr[val[0]])
                    
                    # Initiating Prophet:
                    ml = fbprophet.Prophet()     # Creating an instance of Prophet object
                    ml.fit(df_fr)                    # Fitting prophet object to dataset
                    
                    # Making future datapoints based on forecast duration input (dur):
                    future_points = ml.make_future_dataframe(periods= dur)

                    # Forecasting:
                    forecast = ml.predict(future_points)
                    
                    comp = ml.plot_components(forecast)

                    # Applying Inverse Box-Cox Transform 
                    forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda                                                          x: inv_boxcox(x, lam))
                    
                    # Reshaping historical + forecast
                    future_forecast = forecast.tail(dur)
                    future_forecast = future_forecast[['ds','yhat','yhat_upper','yhat_lower']]
                    future_forecast['type'] = 'Forecast'
                    future_forecast = future_forecast.rename(columns={'ds':'index', 'yhat':val[0]}).set_index('index')
                    df_fr['yhat_upper'] = 0 
                    df_fr['yhat_lower'] = 0 
                    df_fr['type'] = 'Historic'
                    df_fr = df_fr.rename(columns={date[0]:'index'}).set_index('index')
                    df_fr = df_fr[[val[0],'yhat_upper','yhat_lower','type']]
                    df_comb = df_fr.append(future_forecast)
           
                    # Plotting Forecast
                    rng = ['firebrick','grey']
                    dom = ['Historic', 'Forecast']
                    ml_plot = alt.Chart(df_comb.reset_index()).encode(x = alt.X('index:T', axis=alt.Axis(title=                                                                      date[0])),y=val[0],color=alt.Color('type', scale=alt.Scale(domain=dom,                                                        range=rng)),).interactive()
                    st.success('Done')
                    st.subheader("Forecast Components:")
                    st.write(comp)
                    st.subheader("Time Series Plot with Forecast:")
                    st.altair_chart(ml_plot.mark_line().properties(width=800,height=400))   
                    #st.write(fr_plot)

                


if __name__ == '__main__': 
    main()
    
    








