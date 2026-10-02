import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Load pipeline and dataset
pipe = joblib.load('pipeline.joblib')
df = load_breast_cancer(as_frame=True).frame
X = df.drop(columns='target')
y = df['target']
typical = df.groupby('target').mean()

st.title('Breast Tumour Risk Screener')
st.caption('Educational demo only. Not a medical device.')

choice = st.radio('Start from a typical:', ['benign sample', 'malignant sample'], key='sample_choice')
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

st.divider()

# Model Comparison Section
st.subheader("Classifier Performance Comparison")
st.write("Comparing key classifiers across 5-fold cross-validation:")

@st.cache_data
def evaluate_models():
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine (SVM)': make_pipeline(StandardScaler(), SVC(probability=True))
    }
    
    results = []
    for name, model in models.items():
        cv = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'recall'])
        results.append({
            'Model': name,
            'Accuracy (%)': f"{cv['test_accuracy'].mean() * 100:.2f}%",
            'Recall (%)': f"{cv['test_recall'].mean() * 100:.2f}%"
        })
    return pd.DataFrame(results)

results_df = evaluate_models()
st.table(results_df)

st.divider()

# ROC & Precision-Recall Curves
st.subheader("Diagnostic Evaluation Curves")
st.write("Visualizing sensitivity vs. specificity trade-offs on unseen test data:")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

col1, col2 = st.columns(2)

with col1:
    st.write("**ROC Curve**")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax_roc)
    st.pyplot(fig_roc)

with col2:
    st.write("**Precision-Recall Curve**")
    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax_pr)
    st.pyplot(fig_pr)

st.divider()

# Feature Importance Chart
st.subheader("Key Diagnostic Features (Logistic Regression)")
coefs = pd.Series(pipe.named_steps['logisticregression'].coef_[0], index=X.columns)
top_features = coefs.abs().sort_values(ascending=False).head(5)

fig, ax = plt.subplots()
top_features.plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel("Absolute Feature Weight")
st.pyplot(fig)