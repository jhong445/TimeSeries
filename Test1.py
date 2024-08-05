import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller 
import pandas as pd 
import numpy as np 
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.stattools import bds
import pandas_datareader as data
import statsmodels.tsa.stattools as stat

ibm = pd.read_csv("C:\\Users\\Jasper\\Downloads\\IBM.csv")    

ibm['Log Return'] = 100*np.log(ibm['Adj Close']/ibm['Adj Close'].shift(1))
ibm = ibm.drop([0], axis = 0)

plt.plot(ibm['Log Return'],label='Daily Log Return')
plt.legend(loc='best', fontsize='large')
plt.show()

#No Trend
X = ibm['Log Return'].values
result = adfuller(X, maxlag=None, regression='c', autolag='BIC', store=False, regresults=False)
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
	print('\t%s:%.3f'%(key,value))
if result[0] < result [4] ["5%"]:
	print("Reject Ho_ . Time Series is then stationary")
else: 
	print("Failed to Reject Ho_ . Time Series is then non-stationary")

#With trend
result = adfuller(X, maxlag=None, regression='ct', autolag='BIC', store=False, regresults=False)
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
	print('\t%s:%.3f'%(key,value))
if result[0] < result [4] ["5%"]:
	print("Reject Ho_ . Time Series is then stationary")
else: 
	print("Failed to Reject Ho_ . Time Series is then non-stationary")

# itting ARIMA(1,1)
arima=ARIMA(ibm['Log Return'].values,exog=None, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), trend=None, enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=True)
results = arima.fit()
print(results.summary())

#bds test
import statistics
var= statistics.variance(results.resid)
se= var**0.5
std_res=results.resid/se
bds = stat.bds(std_res,max_dim=2, epsilon=None, distance = 1.5)
print('bds_stat, pvalue:{}'.format(bds))

