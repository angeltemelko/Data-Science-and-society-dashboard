"""
Created on Monday October 4 2021
@author: Angel Temelko

"""
from os import name
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px


# Data is read from a local file
stores = pd.read_csv("data2/stores.csv")
features = pd.read_csv("data2/features.csv")
test = pd.read_csv("data2/test.csv")
train = pd.read_csv("data2/train.csv")

#merge features and stores with key 'Stores'
myDataFrame = features.merge(stores, how='inner', on='Store')

# The types of this dataframe are:
pd.DataFrame(myDataFrame.dtypes, columns=['Type'])

#test and train data types
pd.DataFrame(train.dtypes, columns=['Type_Train'])
pd.DataFrame(test.dtypes, columns=['Type_Test'])


#Connvert date from string to date format
myDataFrame.Date = pd.to_datetime(myDataFrame.Date)
train.Date = pd.to_datetime(train.Date)
train['WeekOfYear'] = (train.Date.dt.isocalendar().week)*1.0   
test.Date = pd.to_datetime(test.Date)

#create week and year fields
myDataFrame['Week'] = myDataFrame.Date.dt.isocalendar().week
myDataFrame['Year'] = myDataFrame.Date.dt.isocalendar().year

#trainData and testData joining train and test with myDataFrame
trainData = train.merge(myDataFrame, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
testData = test.merge(myDataFrame, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

del features, train, stores, test

# replace emtpy values with NaN.
trainData = trainData.replace({ "": np.nan, None: np.nan, "%": np.nan, "$": np.nan})

# avareage weekly sales for each year
weeklySales2010 = trainData[trainData.Year == 2010]['Weekly_Sales'].groupby(trainData['Week']).mean()
weeklySales2011 = trainData[trainData.Year == 2011]['Weekly_Sales'].groupby(trainData['Week']).mean()
weeklySales2012 = trainData[trainData.Year == 2012]['Weekly_Sales'].groupby(trainData['Week']).mean()

# figure for avareage weekly sales for each year
weekly_sales_fig = go.Figure()
weekly_sales_fig.add_trace(go.Line(x=weeklySales2010.index,y=weeklySales2010.values,name = '2010'))
weekly_sales_fig.add_trace(go.Line(x=weeklySales2011.index,y=weeklySales2011.values,name = '2011'))
weekly_sales_fig.add_trace(go.Line(x=weeklySales2012.index,y=weeklySales2012.values,name = '2012'))
weekly_sales_fig.update_layout(
    title="Average weekly sales per year",
    xaxis_title="Week",
    yaxis_title="Sales",
    legend_title="Year",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)


#Added easter as holiday in dataset
trainData.loc[(trainData.Year == 2010) & (trainData.Week == 13), 'IsHoliday'] = True
trainData.loc[(trainData.Year == 2011) & (trainData.Week == 16), 'IsHoliday'] = True
trainData.loc[(trainData.Year == 2012) & (trainData.Week == 14), 'IsHoliday'] = True

#Weekly sales mean figure
weekly_sales_mean = trainData['Weekly_Sales'].groupby(trainData['Date']).mean()
weekly_sales_mean_fig = go.Figure()
weekly_sales_mean_fig.add_trace(go.Line(x=weekly_sales_mean.index,y=weekly_sales_mean.values))

# Average Sales per Store
average_sales_store = trainData['Weekly_Sales'].groupby(trainData['Store']).mean()
average_sales_store_fig = go.Figure()
average_sales_store_fig.add_trace(go.Bar(x=average_sales_store.index, y=average_sales_store.values, name='Average Sales per Store', marker_color='#1A0763'))
average_sales_store_fig.update_layout(
    title="Average sales per Store",
    xaxis_title="Store",
    yaxis_title="Sales",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

# Average sales per Department
average_sales_dep = trainData['Weekly_Sales'].groupby(trainData['Dept']).mean()
average_sales_dep_fig = go.Figure()
average_sales_dep_fig.add_trace(go.Bar(x=average_sales_dep.index, y=average_sales_dep.values, name='Average sales per Department', marker_color='#1A0763'))
average_sales_dep_fig.update_layout(
    title="Average sales per Department",
    xaxis_title="Department",
    yaxis_title="Sale",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

# Holiday sales
holiday_sales = trainData.groupby('IsHoliday')['Weekly_Sales'].mean()
holiday_sales_fig = go.Figure()
holiday_sales_fig.add_trace(go.Bar(x=holiday_sales.index, y=holiday_sales.values,name='Holiday sales',marker_color='#1A0763'))
holiday_sales_fig.update_layout(
    title="Sales based on holidays",
    xaxis_title="IsHoliday",
    yaxis_title="Sales",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)


#Metrics for the images
def get_figure_image(name):
    img_width = 1250
    img_height = 900
    scale_factor = 0.5
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )
    # Configure axes
    figure.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    figure.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )
    figure.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=name))
    figure.update_layout(
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return figure

# Adding image instead of loading the graphs, because it takes a lot of time

# Correleation between week of year and sales
plt.figure(figsize=(16,8))
sns.scatterplot(x=trainData.WeekOfYear, y=trainData.Weekly_Sales, hue=trainData.Type,palette="deep", s=80)
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Week of Year', fontsize=20, labelpad=20)
plt.ylabel('Sales', fontsize=20, labelpad=20)
plt.savefig('plots/Correlation between week of year and weekly sales')

# Get image instead of plot
week_of_year_and_sales_fig = get_figure_image(Image.open("plots/Correlation between week of year and weekly sales.png"))

# Correleation between size and sales
plt.figure(figsize=(16,8))
sns.scatterplot(x=trainData.Size, y=trainData.Weekly_Sales, hue=trainData.Type,palette="deep", s=80)
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Size', fontsize=20, labelpad=20)
plt.ylabel('Sales', fontsize=20, labelpad=20)
plt.savefig('plots/Correlation between size and weekly sales.png')

# Get image instead of plot
size_and_sales_fig = get_figure_image(Image.open('plots/Correlation between size and weekly sales.png'))

# Correleation between Temperature and Sales
plt.figure(figsize=(16,8))
sns.scatterplot(x=trainData.Temperature, y=trainData.Weekly_Sales, hue=trainData.Type,palette="deep", s=80)
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Temperature', fontsize=20, labelpad=20)
plt.ylabel('Sales', fontsize=20, labelpad=20)
plt.savefig('plots/Correlation between temperature and weekly sales')

# Get image instead of plot
size_and_temperature_fig = get_figure_image(Image.open('plots/Correlation between temperature and weekly sales.png'))

# Correleation between Fuel price and Sales
plt.figure(figsize=(16,8))
sns.scatterplot(x=trainData.Fuel_Price, y=trainData.Weekly_Sales, hue=trainData.Type,palette="deep", s=80)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Fuel Price', fontsize=20, labelpad=20)
plt.ylabel('Sales', fontsize=20, labelpad=20)
plt.savefig('plots/Correlation between Fuel price and weekly sales.png')

# Get image instead of plot
fuel_and_sales_fig = get_figure_image(Image.open('plots/Correlation between Fuel price and weekly sales.png'))

# Correleation between CPI and Sales
plt.figure(figsize=(16,8))
sns.scatterplot(x=trainData.CPI, y=trainData.Weekly_Sales, hue=trainData.Type,palette="deep", s=80);
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('CPI', fontsize=20, labelpad=20)
plt.ylabel('Sales', fontsize=20, labelpad=20);
plt.savefig('plots/Correlation between CPI and weekly sales.png')

# Get image instead of plot
cpi_and_sales_fig = get_figure_image(Image.open('plots/Correlation between CPI and weekly sales.png'))

# Correleation between Unemployment and Sales
plt.figure(figsize=(16,8))
sns.scatterplot(x=trainData.Unemployment, y=trainData.Weekly_Sales, hue=trainData.Type,palette='dark', s=80);
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Unemployment', fontsize=20, labelpad=20)
plt.ylabel('Sales', fontsize=20, labelpad=20);
plt.savefig('plots/Correlation Unemployment size and weekly sales.png')

# Get image instead of plot
unemployment_and_sales_fig = get_figure_image(Image.open('plots/Correlation Unemployment size and weekly sales.png'))

#Get the number of stores
typecounts = trainData.Type.value_counts().to_dict()
pieFigData = pd.DataFrame(list(typecounts.items()), columns=['Store_Type', 'Counts'])