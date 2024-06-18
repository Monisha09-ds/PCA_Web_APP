import streamlit as st 
import pandas as pd
import numpy as np 
import plotly.express as px  

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA 

st.title("Streamlit Priciple Component Analysis Demo")
st.subheader('RawData')
st.subheader('Credit Card Information')

data = pd.read_csv('german_credit_data.csv')
