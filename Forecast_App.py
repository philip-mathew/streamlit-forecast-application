############################################################################
###                                                                      ###
##  Time Series Forecast Application on Streamlit using Facebook Prophet  ##
###                                                                      ###
##                          Built by Philip Mathew                        ##                          
############################################################################



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fbprophet 
import altair as alt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import base64

# CSV Download link generation function
def download_link(object_to_download, download_filename, download_link_text):
    
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv()

    # strings <-> bytes conversions 
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# main function
def main():
    
    # title & header
    st.title("Time Series Forecast Application")
    st.markdown("Built by [Philip Mathew](https://github.com/philip-mathew)")
    st.write("Data Science web application for automated forecasting of univariate time-series data.")
    st.write("Univariate time-series dataset (csv) can be uploaded below and forecast period can be inputed for automated forecasting using facebook prophet library and box-cox data transform.")
    st.markdown("Prophet forecasts time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is also robust to missing data and shifts in the trend, and typically handles outliers well.")
    st.markdown("---")
    st.header('Input Time Series Dataset')
    
    # Streamlit file upload - csv only
    data_file = st.file_uploader("Upload CSV with two columns; Date/Time & Variable:",type=['csv'])
    df_comb = None
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
            st.write(df[val[0]].describe())
            ca = alt.Chart(df.reset_index()).encode(x = alt.X('index:T', axis=alt.Axis(title=                       
                                                                  date[0])),y=val[0]).interactive()
            st.subheader("Time Series Plot:")
            st.altair_chart(ca.mark_line(color='firebrick').properties(width=1100,height=400))    
    
    st.markdown("---")
    
    # Forecast frequency selection
    st.header('Select Forecast Frequency')
    freq = st.selectbox('Select the frequency of forecast:', ['Day', 'Week', 'Month', 'Quarter', 'Year', '1 Second', '1 Minute',                                                               '1 Hour'])
    if freq == 'Day':
        freq = 'D'
    elif freq == 'Week':
        freq = 'W' 
    elif freq == 'Month':
        freq = 'M'
    elif freq == 'Quarter':
        freq = 'Q'
    elif freq == 'Year':
        freq = 'Y'
    elif freq == '1 Second':
        freq = 1     
    elif freq == '1 Minute':
        freq = 60
    elif freq == '1 Hour':
        freq = 3600 
    
    st.markdown("---")
    
    # Forecast duration selection
    st.header('Select Forecast Period')
    dur = st.number_input('Enter number of periods (of frequency) to forecast in future:',  min_value = 1, step = 1)
    
    st.markdown("---")

    # Forecast Button
    if st.button("Forecast"):
            if data_file is not None:  
                st.spinner()
                with st.spinner(text='Prophet is forecasting....'):
                    
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
                    future_points = ml.make_future_dataframe(periods= dur, freq = freq)

                    # Forecasting:
                    forecast = ml.predict(future_points)
                    
                    # Components of forecast 
                    comp = ml.plot_components(forecast)

                    # Applying Inverse Box-Cox Transform 
                    forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda                                                          x: inv_boxcox(x, lam))
                    
                    # Reshaping historical + forecast:
                    future_forecast = forecast.tail(dur)
                    future_forecast = future_forecast[['ds','yhat','yhat_upper','yhat_lower']]
                    future_forecast['Type'] = 'Forecast'
                    future_forecast = future_forecast.rename(columns={'ds':'index', 'yhat':val[0]}).set_index('index')
                    df_fr['yhat_upper'] = 0 
                    df_fr['yhat_lower'] = 0 
                    df_fr['Type'] = 'Historic'
                    df_fr = df_fr.rename(columns={date[0]:'index'}).set_index('index')
                    df_fr = df_fr[[val[0],'yhat_upper','yhat_lower','Type']]
                    df_comb = df_fr.append(future_forecast)
           
                    # Plotting Forecast
                    rng = ['firebrick', '#4267B2']
                    dom = ['Historic', 'Forecast']
                    base = alt.Chart(df_comb.reset_index()).encode(x = alt.X('index:T', axis=alt.Axis(title=                                                                      date[0])))  
                    ml_plot = base.mark_line(interpolate='monotone').encode(y=val[0],color=alt.Color('Type',                                                                         scale=alt.Scale(domain=dom, range=rng)),).interactive()
                    area = base.mark_area(opacity=0.5,color='grey').encode(alt.Y('yhat_upper'),                                                                     alt.Y2('yhat_lower')).interactive()
                    
                    # Processing complete
                    st.success('Done')
                    st.markdown("---")
                    st.header('Time Series Forecast')
                    
                    # Plotting
                    st.subheader("Forecast Components:")
                    st.write(comp)
                    st.subheader("Time Series Plot with Forecast:")
                    st.altair_chart(alt.layer(ml_plot,area).properties(width=1100,height=400).resolve_scale(y = 'independent'))
    
                    # Download link for forecast (csv)
                    tmp_download_link = download_link(df_comb, 'Data_w_forecast.csv', 'Click here to download your forecast!')
                    st.subheader("Download Forecast (csv file):")
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    st.markdown("---")


                
if __name__ == '__main__': 
    main()
    
