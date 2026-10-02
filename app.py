import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

pipe = joblib.load('pipeline.joblib')

df = load_breast_cancer(as_frame=True).frame
X = df.drop(columns='target')
typical = df.groupby('target').mean()

st.title('Breast Tumour Risk Screener')
st.caption('Educational demo only. Not a medical device.')

choice = st.radio('Start from a typical:', ['benign sample', 'malignant sample'])
start = typical.loc[1 if choice == 'benign sample' else 0]

values = {}
for col in X.columns:
    values[col] = st.sidebar.slider(
        col, 
        float(X[col].min()), 
        float(X[col].max()), 
        float(start[col]), 
        key=f'{col}-{choice}'
    )

if st.button('Predict'):
    p_malignant = 1 - pipe.predict_proba(pd.DataFrame([values]))[0][1]
    
    st.metric('Estimated chance of malignancy', f'{p_malignant:.0%}')
    
    if p_malignant > 0.30:
        st.error('Flag for follow-up (specialist review recommended)')
    else:
        st.success('Low estimated risk')

import matplotlib.pyplot as plt

st.subheader("Key Diagnostic Features")
# Extract model coefficients/weights
coefs = pd.Series(pipe.named_steps['logisticregression'].coef_[0], index=X.columns)
top_features = coefs.abs().sort_values(ascending=False).head(5)

fig, ax = plt.subplots()
top_features.plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel("Absolute Feature Weight")
st.pyplot(fig)