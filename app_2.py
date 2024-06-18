import numpy as np
import pandas as pd
import streamlit as st 
import yfinance as yf
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import date,timedelta 
import statsmodels.api as sm 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


app_name="Stock Market Prediction App"
st.title(app_name)
st.image("https://images.search.yahoo.com/images/view;_ylt=Awr91xXxGJtlFa8GG1OJzbkF;_ylu=c2VjA3NyBHNsawNpbWcEb2lkA2JhOGNiOWExMTIzZjhmMGQ1OTQ3Y2Y3YzcwYzM3OTBkBGdwb3MDMTgEaXQDYmluZw--?back=https%3A%2F%2Fimages.search.yahoo.com%2Fsearch%2Fimages%3Fp%3Dstock%2Bprediction%2Bimage%26ei%3DUTF-8%26type%3DE210US885G0%26fr%3Dmcafee%26fr2%3Dp%253As%252Cv%253Ai%252Cm%253Asb-top%26tab%3Dorganic%26ri%3D18&w=626&h=417&imgurl=miro.medium.com%2Fmax%2F626%2F0%2ASaNg8uUaKCMQSS5g.jpg&rurl=https%3A%2F%2Fmedium.com%2F%40kala.shagun%2Fstock-market-prediction-using-news-sentiments-f9101e5ee1f4&size=27.1KB&p=stock+prediction+image&oid=ba8cb9a1123f8f0d5947cf7c70c3790d&fr2=p%3As%2Cv%3Ai%2Cm%3Asb-top&fr=mcafee&tt=Stock+Market+Prediction+using+News+Sentiments-+I+%7C+by+Shagun+Kala+%7C+Medium&b=0&ni=90&no=18&ts=&tab=organic&sigr=pSWz_xrgeKx6&sigb=ZAEVQE4K5lzc&sigi=jObXrHh9KyAr&sigt=tkMLH6Zr3RrV&.crumb=SIb5aCqyx14&fr=mcafee&fr2=p%3As%2Cv%3Ai%2Cm%3Asb-top&type=E210US885G0")
#Take input from theuser of app about the start and end date
start_date = st.sidebar.date_input('Start Date',date(2023,12,1))
end_date = st.sidebar.date_input('End Date',date.today())

#add ticker symbol list
ticker_list = ['AAPL','MSFT','GOOGLE','FB','TSLA','NVDA','ADBE','PYPL','INTC','CHCSA','NFLX','PEP']
ticker = st.sidebar.selectbox("Select the company",ticker_list)

#fetch the data
data = yf.download(ticker,start_date,end_date)
data.insert(0,'Date',date.index,True)
data.reset_index(drop=True,inplace=True)
st.write(data)

#plot the data 
st.header("Data Visualization")
st.subheader("Plot of the data")
fig = px.line(data,x='Date',y=data.columns,title ="Closing price of the stock")
st.plotly_chart(fig)


#add column box to select column from data
column =st.selectbox("Select the column used for forecasting",data.columns[1:])

data = data[['Date',column]]
st.write('selected Data')
st.write(data)

#ADF test check stationarity
st.header("Is data stationary??")
st.write(adfuller(data[column])[1]<0.05)

#lets Decompose the data 
st.header("Decomposition of the data")
decomposition = seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

# Make a time series plots
st.write("##Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend,title="Trend",width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal,title="Seasonal",width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid,title="Residuals",width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Red'),line_dash='dot')

#Lets run the model
#user input for the three parameters
p = st.slider('Select the value of p',0,5,2)
d = st.slider('Select the value of p',0,5,1)
q = st.slider('Select the value of p',0,5,2)
seasonal_order = st.number_input('Select the value of seasonal p',0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

#print model summary
st.header("Model Summary")
st.write(model.summary())
st.write("---")

#Forecasting(predict the future value)
forecast_period = st.number_input('Select the number of days to forecast',1,365,10)
prediction = model.get_prediction(start=len(data),end=len(data)+forecast_period)

#show prediction
prediction = prediction.predicted_mean
st.write(prediction)

#add index to the predictions
prediction.index = pd.date_range(start = end_date,periods = len(prediction),freq="D")
df = pd.DataFrame(prediction)
prediction.insert(0,'Date',prediction.index,True)
st.write("Predicted Data",prediction)
st.write("Actual Data",data)
st.write('-----')

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines',name="Actual",line=dict(color='blue')))
fig.add_trace(go.Scatter(x=prediction["Date"],y=prediction["predicted_mean"],mode='lines',name="Predicted",line=dict(color='red')))
#fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines',name="Actual",line=dict(color='blue')))
fig.update_layout(title="Forecast vs Actual", xaxis_title ="Date",yaxis_title="Price",width=1200,height=400)
st.plotly_chart(fig)

#Add buttons to show and hide separate plots

show_plots = False
if st.button("Show separate plot"):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],mode='lines',name="Actual",line=dict(color='blue')))