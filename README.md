# Web-App to Forecast Univariate Time-Series Data

#### Univariate time-series dataset (csv) can be uploaded to web-app, along with time period frequency and number of periods to forecast for automated forecasting using facebook prophet library.

#### Prophet forecasts time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is also robust to missing data and shifts in the trend, and typically handles outliers well.

https://facebook.github.io/prophet/

Web-App can be accessed at https://time-series-forecasting.herokuapp.com/

## Steps to use Web-App:

1. Upload csv file of univariate time series dataset for forecasting.
2. Select time series frequency (Day, Week, Month, Quaterly, Year, Second, Minute or Hour).
3. Select number of future periods (of frequency) to forecast.
4. Click on forecast button to generate forecast.
5. Click on link at bottom to download csv of (history + forecast).  
