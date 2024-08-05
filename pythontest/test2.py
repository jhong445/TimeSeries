#libraries
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller 
import pandas as pd 
import numpy as np 
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.stattools import bds
import pandas_datareader as data
import statsmodels.tsa.stattools as stat
import yfinance as yf
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
import itertools
from statsmodels.graphics.tsaplots import plot_predict
from pandas import datetime
from datetime import date, datetime, timedelta


#ibm = yf.download("IBM", start='2015-9-1', end='2021-6-30')
ibm = pd.read_csv("C:\\Users\\Jasper\\Downloads\\IBM.csv",
                  parse_dates=['Date'], index_col='Date', header=0,sep=",")
ibm['Log Return'] = 100*np.log(ibm['Adj Close']/ibm['Adj Close'].shift(1))
ibm = ibm.tail(-1)
ibm.drop(['Open', 'Close', 'Adj Close', 'Volume', 'High', 'Low'], axis = 1, inplace = True )

#creating test and training sets
points = 1600
train, test = ibm.iloc[:points], ibm.iloc[points:]
'''
plt.plot(train.index, train['Log Return'], label='Training Data')
plt.plot(test.index,test['Log Return'], label='Testing Data')
plt.legend()

plt.show()
'''
#finding best arima
'''
model = auto_arima(train, start_p=0, start_q=0,
                      test='adf',
                      max_p=5, max_q=5,
                      m=1,             
                      d=1,          
                      seasonal=False,   
                      start_P=0, 
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
'''
# fitting model
arima = ARIMA(endog=train['Log Return'].values, order=(5, 1, 0),
              seasonal_order=(0, 0, 0, 0), trend=None,
              enforce_stationarity=True, enforce_invertibility=True,
              concentrate_scale=True)
results = arima.fit()
print(results.summary())

#plotting model forecasts
y = test.copy()
y = y.reset_index()
y['ARIMA'] = results.predict(start = points, end = ibm.shape[0]-1, dynamic=False)
'''
plt.figure(figsize=(16,8))
plt.plot(train.index, train['Log Return'], label='In-sample data')
plt.plot(test.index, test['Log Return'], label='Out-sample data')
plt.plot(y['Date'], y['ARIMA'], label='Arima model')
plt.legend(loc='best')
plt.show()
'''

rmsfear = sqrt(mean_squared_error(test['Log Return'], y['ARIMA']))
print('Rmse is:' + str(rmsfear))

#forecasting into the future
forecast = results.get_forecast(steps=100)
sdate = date(2023,3,23)   # start date
edate = sdate + timedelta(days=100)   # end date

dates = pd.date_range(sdate,edate-timedelta(days=1),freq='d')

mean_forecast = forecast.predicted_mean
ci = forecast.conf_int()
plt.plot(train.index, train['Log Return'], label='In-sample data')
plt.plot(test.index, test['Log Return'], label='Out-sample data')
plt.plot(y['Date'], y['ARIMA'], label='Arima model')
plt.plot(dates, mean_forecast,
         color= 'red',
         label= 'Forecast')
plt.legend(loc='best')
plt.show()
