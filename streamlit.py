import streamlit as st
import pandas as pd 
import pickle

with open('train_model.pkl','rb') as file:
    model = pickle.load(file)
    
st.title('Salary Prediction App')
st.header('Upload Data for Prediction')    
uploaded_file = st.file_uploader('sample_data', type='csv')
if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        data.columns = data.columns.str.upper()
        test = data[['AGE','PAST EXP']]
        test = test.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
        test.columns = test.columns.str.lower()
        test = test.values
        if test is not None:
            prediction = model.predict(test)
            data['PREDICTED_SALARY'] = prediction
        else:
            st.write("Error processing data. Please check the input file.")    
        
        st.write(data)