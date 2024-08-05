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

#ibm = yf.download("IBM", start='2015-9-1', end='2021-6-30')
ibm = pd.read_csv("C:\\Users\\Jasper\\Downloads\\IBM.csv")
ibm['Log Return'] = 100*np.log(ibm['Adj Close']/ibm['Adj Close'].shift(1))
ibm = ibm.drop([0], axis = 0)

#test if data is stationary
x = ibm['Log Return'].values
result = adfuller(x, maxlag=None, regression='c', autolag='BIC', store=False, regresults=False)
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
	print('\t%s:%.3f'%(key,value))
if result[0] < result [4] ["5%"]:
	print("Reject Ho_ . Time Series is then stationary")
else: 
	print("Failed to Reject Ho_ . Time Series is then non-stationary")

#splitting data into training and testing sets
train = ibm[(ibm['Date'] < '2020-07-01')]
test = ibm[(ibm['Date'] > '2020-07-01')]

plt.plot(train.index, train['Log Return'], label='Training Data')
plt.plot(test.index,test['Log Return'], label='Testing Data')
plt.show()

#arima 
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)

model.summary()
