import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import pairwise_distances_argmin_min
with open('acmodel.pkl','rb') as file:
    acmodel=pickle.load(file)
with open('scalar_obj.pkl','rb') as file:
    obj=pickle.load(file)
d={0:'Average income with low spending',
1:'Low income with average spending',
2:'High income with low spending',
3:'High income with high spending'}
df_str=pd.read_csv('dataset.csv')
df_sc=pd.read_csv('original_df.csv')
st.set_page_config(page_title='Customer Segmentation')
st.header('Welcome to Customer segmentation report!!!')
with st.container(border=True):
    c1,c2=st.columns(2)
    gen=c1.radio('Gender',options=df_str['Gender'].unique())
    age=c2.slider('Age',min_value=0,max_value=100)
    Anuin=c1.number_input('Annual Income')
    spensc=c2.slider('Spending Score',min_value=0,max_value=100)
    gender=list(df_str['Gender'].unique())
    gender.sort()
    in_vals=[[gender.index(gen),age,Anuin,spensc]]
    in_vals=obj.transform(in_vals)
    df_new=pd.DataFrame(in_vals,columns=['Gender','Age','Annual Income','Spending Score'])
    if st.button('Submit'):
        df_sc=df_sc.drop('Unnamed: 0',axis=1)
        new,_=pairwise_distances_argmin_min(df_new,df_sc)
        label=acmodel.labels_[new]
        st.subheader(d[label[0]])



