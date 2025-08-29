import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import time


st.header('Body Fat Prediction Using Machine Learning')
st.image('fatpred.jpg')

data = '''
Body Fat Prediction using Machine Learning 🧠📊

Maintaining healthy body fat is crucial for fitness and health.  
In this project, I analyzed a **Body Fat dataset** with preprocessing and trained multiple ML models.  
Finally, a **Random Forest Regressor** model was chosen for accurate predictions.  

### Algorithms Tried:
- Linear Regression
- Decision Tree
- Random Forest

Final model: **Random Forest** ✅
'''

st.markdown(data)

st.image('bodyfat.jpg')  


with open('fat_prediction.pkl','rb') as f:
    model = pickle.load(f)


url = "https://raw.githubusercontent.com/prathimacode-hub/ML-ProjectKart/main/Body%20Fat%20Prediction/Dataset/bodyfat.csv"
df = pd.read_csv(url)


st.sidebar.header('Select Features to Predict Body Fat')
st.sidebar.image('https://c.ndtvimg.com/2020-05/7bqgah7o_obesity-650-_625x300_14_May_20.jpg')

all_values = []


for col in df.columns:
    if col.lower() == "bodyfat":  
        continue
    
    min_value, max_value = df[col].agg(['min','max'])

   
    try:
        var = st.sidebar.slider(
            f'Select {col} value', 
            float(min_value), float(max_value), 
            float(df[col].mean())
        )
        all_values.append(var)
    except:
        pass

final_value = [all_values]


ans = model.predict(final_value)[0]

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Body Fat...') 

place = st.empty()
place.image('https://cdn-icons-gif.flaticon.com/19003/19003858.gif',width = 200)

for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

placeholder.empty()
place.empty()
st.success(f'Predicted Body Fat: **{ans:.2f}%** ✅')
progress_bar = st.progress(0)


st.markdown('Designed by: **Sumeet Singh & Mohd Ayan**')