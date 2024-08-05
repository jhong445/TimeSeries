#Forecasting
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


#ibm = yf.download("IBM", start='2015-9-1', end='2021-6-30')
ibm = pd.read_csv("C:\\Users\\Jasper\\Downloads\\IBM.csv")


train = ibm[(ibm['Date'] < '2020-07-01')]
test = ibm[(ibm['Date'] > '2020-07-01')]

plt.plot(train.index, train['Adj Close'], label='In-Sample Data')
plt.plot(test.index,test['Adj Close'], label='Out of Sample Data')

y = train['Adj Close']

#ARIMA model
ARMAmodel = ARIMA(y, order = (1, 1, 1))
ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["ARIMA Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["ARIMA Predictions"] 

plt.plot(y_pred_out, label = 'ARIMA Predictions')
plt.legend()


#SARMIA model
SARMAmodel = SARIMAX(y, order=(2, 2, 2), seasonal_order=(0, 0, 0, 0))
SARMAmodel = SARMAmodel.fit(disp=False)

y_pred = SARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["SARIMA Predictions"] = SARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["SARIMA Predictions"] 

plt.plot(y_pred_out, label = 'SARIMA Predictions')
plt.legend()
