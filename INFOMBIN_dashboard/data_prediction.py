"""
Created on Monday October 20 2021
@author: Angel Temelko
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import rmse 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import calendar
import itertools
from plotly.tools import mpl_to_plotly
import plotly.graph_objs as go
import plotly.tools as tls
from data_preparation_descriptive_part import trainData,get_figure_image,testData
import warnings
from PIL import Image
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

df=trainData[['Store','Dept','Date','Weekly_Sales','IsHoliday']]

# Encode isholidy column into numeric values so that we can do numeric calculation
df['IsHoliday01'] = df['IsHoliday'].apply(lambda x: '1' if x == True else '0') 

#set date index
df.index=df.Date

#creating new copy of train dataset for analysis
ts=df.copy() 

ts.index=(ts.Date)
#ts.index

#extract month and year column from date and place them as separate column
ts['Month'] = pd.DatetimeIndex(ts['Date']).month
ts['Month'] = ts['Month'].apply(lambda x: calendar.month_abbr[x]) 
ts['Year'] = pd.DatetimeIndex(ts['Date']).year

ts.head(10) 

#Extract rows of store 1
ts_1=ts[ts.Store==1] 

#Iterate all dept of store 1
# plt.figure(figsize=(10,10))
for i in ts_1.Dept.unique(): 
    ts_1[ts_1['Dept']==i]['Weekly_Sales'].plot()

#Extract rows of dept 1 of store 1 
ts_1=ts_1[ts_1.Dept==1]

# Plot the weekly sales
# ts_1['Weekly_Sales'].plot() 

#Creating pivot table of monthly sales , to see monthly sales frequency of every year.
monthly_sales_data = pd.pivot_table(ts_1, values = "Weekly_Sales", columns = "Year", index = "Month")

#Reindex the dataset and set the index according to months
monthly_sales_data = monthly_sales_data.reindex(index = ['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#monthly_sales_data 

# # Plotting monthly sales frequency
# montmonthly_sales_data_fig = monthly_sales_data.plot() 

# Finding yearly sales data.
yearly_sales_data = monthly_sales_data.T


# yearly_sales_data_fig = yearly_sales_data.plot() 

# Finding mean of weekly sales data by resampling to momthly frequency
# Aggregating full dataset into monthly sales data.
monthly_sales = df['Weekly_Sales'].resample('M').mean() 



# plt.figure(figsize=(15,8))
# plt.ylabel('Sales',fontsize=20)
# plt.xlabel('Month',fontsize=20)
# plt.title('Sales per Month',fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# monthly_sales.plot()
# plt.plot(monthly_sales)
# plt.show()

# plotly_fig = mpl_to_plotly(omg)

#Defining a function to test that training data is stationery or not.
'''
Null Hypothesis: The time series dataset is not stationary.
Alternate Hypothesis: The time series dataset is stationary.
'''

#define function for ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    # print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    # print (dfoutput)

'''
Test for stationarity: If the test statistic is less than the critical value, we can reject the null hypothesis (that means the series is stationary). When the test statistic is greater than the critical value, we fail to reject the null hypothesis (which means the series is not stationary).
'''

adf_test(monthly_sales)

#Ploting seasonal_decomposional plot to view trend and seasonality of the data

figurce = plt.figure()
decomposition_plot = figurce.add_subplot(111)
decomposition_plot = sm.tsa.seasonal_decompose(monthly_sales, model='additive')
#decomposition_plot.plot()
plotly_fig_4 = mpl_to_plotly(figurce)

#plt.show()

#plotly_fig = 2 # = mpl_to_plotly(decomposition_plot)


# Creating ARIMA model
# Creating matrix of all possible combination of the parameter
a = b = c = range(0, 2)
abc = list(itertools.product(a, b, c))
seasonal_abc = [(x[0], x[1], x[2], 12) for x in list(itertools.product(a,b,c))]  
# print('The combinations for Seasonal ARIMA Model are following:')
# print('SARIMAX: {} x {}'.format(abc[1], seasonal_abc[1]))
# print('SARIMAX: {} x {}'.format(abc[1], seasonal_abc[2]))
# print('SARIMAX: {} x {}'.format(abc[2], seasonal_abc[3]))
# print('SARIMAX: {} x {}'.format(abc[2], seasonal_abc[4]))

# Finding optimal set of parameters for efficient parameters tuning


for parameters in abc:
    for param_seasonal in seasonal_abc:
        try:
            model = sm.tsa.statespace.SARIMAX(monthly_sales,order=parameters,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = model.fit(disp=0)
            # print('ARIMA{}x{}12 - AIC:{}'.format(parameters, param_seasonal, results.aic))
        except:
            continue

#The combination of ARIMA (0, 1, 1)x(0, 1, 1, 12) has the lowest value of AIC = 84.074. Hence, ARIMA (0, 1, 1)x(0, 1, 1, 12) is be used as the optimal solution.

# (0, 1, 1)x(0, 1, 1, 12)
#Fitting The model
opt_model = sm.tsa.statespace.SARIMAX(monthly_sales,order=(0, 1, 1),seasonal_order=(0, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)
result = opt_model.fit(disp=0)
# print(result.summary().tables[1])
#result.summary()

# Forecasting the sales against true parameters

#Get Prediction
pred = result.get_prediction(start=pd.to_datetime('2010-02-28'), dynamic=False)
pred = result.get_prediction(dynamic=False)
#Confidence Interval
pred_ci = pred.conf_int()
#Actual Data
fig = plt.figure()
ax = fig.add_subplot()
ax = monthly_sales['2010-02-28':].plot(label='Actual', figsize=(6.3, 4.5))
predictions=pred.predicted_mean
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.5, figsize=(6.3, 4.5))
ax.fill_between(pred_ci.index,
               pred_ci.iloc[:, 0],
               pred_ci.iloc[:, 1], color='b', alpha=.2)
ax.set_xlabel('Timeframe', fontsize = 13)
ax.set_ylabel('Sales', fontsize = 13)
ax.set_title('Timeframe Vs Sales', fontsize = 15)
ax.set_xlim([datetime.date(2010, 2, 28), datetime.date(2012, 10, 31)])
#plt.legend()
#plt.show()
plotly_fig = mpl_to_plotly(fig)



#From this plot, we can clearly observe that the forecasted line is almost the same as our true line. This conclude that our arima model is perfactly tuned.


#Calculating MSE and RMSE for checking accuracy of our model.
# print('Mean Squared Error:',mean_squared_error(monthly_sales['2010-02-28':],pred.predicted_mean))
# print('Root Mean Squared Error:',np.sqrt(mean_squared_error(monthly_sales['2010-02-28':],pred.predicted_mean)))

test=trainData.copy()
test.index=pd.to_datetime(test.Date)

#Test Dataset
monthly_test_data=test.resample('M').mean()



#Defining a Function for calculating mean_absolute_percentage_error 
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculating the errors
# print('Mean Forecast Error : ',np.mean([monthly_sales['2010-03-31':][i]-predictions['2010-03-31':][i] for i in range(len(monthly_sales['2010-03-31':]))]))
# print('Mean Absolute Error: ',mean_absolute_error(monthly_sales['2010-03-31':],predictions['2010-03-31':]))
# print('Mean Squared Error:',mean_squared_error(monthly_sales['2010-03-31':],predictions['2010-03-31':]))
# print('Root Mean Squared Error:',rmse(monthly_sales['2010-03-31':],predictions['2010-03-31':]))
# print('Mean absolute percentage error :',mean_absolute_percentage_error(monthly_sales['2010-03-31':],predictions['2010-03-31':]))


#have an copy of test data. We don't want to use the original copy. 
test=testData.copy()

# make index as date to do the analysis 
test.index=pd.to_datetime(test.Date)

#Resample for having monthly sales as we are forecasting for monyhly sales here.
monthly_test_data=test.resample('M').mean()

#Total Months
#len(monthly_test_data)

monthly_sales
#Forecast 9 steps means forecast 9 month ahead
plt.clf()
pred_uc = result.get_forecast(steps=len(monthly_test_data))
pred_ci = pred_uc.conf_int()
ax = monthly_sales.plot(label='Observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.25)
ax.set_xlabel('Timeline', fontsize = 13)
ax.set_ylabel('Sales', fontsize = 13)
ax.set_title('Timeline Vs Sales', fontsize = 15)
plt.legend()
#plt.show()
plt.savefig("plots/MonthlyForecast.png")
#plt.show()
#plotly_fig_second = mpl_to_plotly(fig2)

#adding the image to the plot
monthly_sales_fig_pred = get_figure_image(Image.open("plots/MonthlyForecast.png"))

monthly_test_data['monthly_sale'] = pred_uc.predicted_mean

#Predicted Monthly Sales  
pred_monthly_sales = monthly_test_data.reset_index()[['Date', 'monthly_sale']]  
#pred_monthly_sales  

# Export Monthly Sales as CSV
pred_monthly_sales.to_csv('Predicted_monthly.csv', index = False)

'''
In previous, we create arima model for monthly sales forecasting. Now we will create another arima model for weekly sales forecasting of store 1 and dept 1.
'''

# Ploting Weekly sales data
weekly_sales=pd.DataFrame(ts_1['Weekly_Sales'])
plottingWeeklySales = weekly_sales.plot()


#Test for stationarity
adf_test(weekly_sales)

'''
If the test statistic is less than the critical value, we can reject the null hypothesis (that means the series is stationary). When the test statistic is greater than the critical value, we fail to reject the null hypothesis (which means the series is not stationary).
There the test statistic > critical value, which implies that the series is not stationary.
'''

#Converting time series data to a logarithmic scale reduces the variability of the data.
weekly_sales['Weekly_sales_log'] =np.log(weekly_sales['Weekly_Sales'])
weekly_sales['Weekly_sales_log'].dropna().plot() 

# Droping Null Values
weekly_sales.dropna(inplace=True)

# Decomposition plot of weekly timeseries data set.
decomposition_plot = sm.tsa.seasonal_decompose(weekly_sales['Weekly_sales_log'], model='additive')
#decomposition_plot.plot()
#plt.show()

#Creating matrix of all possible combination of the parameter
for parameters in abc:
    for param_seasonal in seasonal_abc:
        try:
            model = sm.tsa.statespace.SARIMAX(weekly_sales['Weekly_sales_log'],order=parameters,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = model.fit(disp=0)
            #print('ARIMA{}x{}12 - AIC:{}'.format(parameters, param_seasonal, results.aic))
        except:
            continue

#The combination of ARIMA (1, 0, 1)x(1, 1, 1, 12) has the lowest value of AIC = 2474.5402229445185. Hence, ARIMA (1, 0, 1)x(1, 1, 1, 12) is be used as the optimal solution

# Define the model with the optimal parameter
opt_model = sm.tsa.statespace.SARIMAX(weekly_sales['Weekly_sales_log'],order=(1, 0, 1),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)

#Fit the model
result = opt_model.fit(disp=0)
#print(result.summary().tables[1])
#result.summary()

#Warnings:
#[1] Covariance matrix calculated using the outer product of gradients (complex-step).
#[2] Covariance matrix is singular or near-singular, with condition number 3.56e+33. Standard errors may be unstable.

fig3 = plt.figure()
pred = result.get_prediction(start=pd.to_datetime('2010-02-12'), dynamic=False)
pred = result.get_prediction(dynamic=False)
ax = weekly_sales[12:]['Weekly_sales_log'].plot(label='Actual')
weekly_sales['forecast']=pred.predicted_mean
weekly_sales[12:]['forecast'].plot(label='forecast')
ax.set_xlabel('Timeframe', fontsize = 13)
ax.set_ylabel('Sales', fontsize = 13)
ax.set_title('Timeframe Vs Sales', fontsize = 15)
plt.legend()
plt.show()
# plt.legend()
# plt.show()

plotly_fig_third = mpl_to_plotly(fig3)


#Evaluation
# print('Mean Forecast Error:',np.mean([np.array(weekly_sales['2010-04-30 ':]['Weekly_Sales'])[i]-np.exp(pred.predicted_mean['2010-04-30 ':])[i] for i in range(len(weekly_sales['2010-04-30':]))]))
# print('Mean Absolute Error:',mean_absolute_error(weekly_sales['2010-04-30 ':]['Weekly_Sales'],np.exp(pred.predicted_mean['2010-04-30':])))
# print('Mean Squared Error:',mean_squared_error(weekly_sales['2010-04-30 ':]['Weekly_Sales'],np.exp(pred.predicted_mean['2010-04-30':])))
# print('Root Mean Squared Error:',rmse(weekly_sales['2010-04-30 ':]['Weekly_Sales'],np.exp(pred.predicted_mean['2010-04-30':])))
# print('Mean Absolute Percentage Error:',mean_absolute_percentage_error(weekly_sales['2010-04-30 ':]['Weekly_Sales'],np.exp(pred.predicted_mean['2010-04-30':])))


fig6 = plt.figure()
ax = fig6
weekly_test_data=test[(test.Store==1) & (test.Dept==1)]
#len(weekly_test_data)
pred_uc = result.get_forecast(steps=len(weekly_test_data))
weekly_sales=pd.concat([weekly_sales,(pred_uc.predicted_mean)], axis=1)
weekly_test_data['Predicted_Weekly_Sales']=np.exp(pred_uc.predicted_mean)
#Weekly Sales Prediction Plot
ax = weekly_sales.Weekly_Sales.plot(label='Observed', figsize=(14, 7))
# weekly_sales.rename(columns = {'predicted_mean': 'predicted_weekly_sales'}, inplace = True)
weekly_test_data['Predicted_Weekly_Sales'].plot(ax=ax, label='Forecast')
plt.legend()
plt.savefig("plots/weeklyforecast.png")
plotly_fig_six = mpl_to_plotly(fig6)

weekly_sales_fig_pred = get_figure_image(Image.open("plots/weeklyforecast.png"))

#plt.legend()
#plt.show()

#Predicted Weekly Sales
pred_week_sale  = weekly_test_data.reset_index(drop = True)[['Date', 'Predicted_Weekly_Sales']]
#pred_week_sale  

pred_week_sale.to_csv('Pred_week_sales.csv', index = False, mode='w+')
