import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Page Config
st.set_page_config(page_title="Breast Tumour Risk Screener", layout="wide")

# Custom CSS for Background & Aesthetics
st.markdown("""
    <style>
    /* Main app background gradient */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
        color: #f8fafc;
    }
    /* Style cards and container boxes */
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    </style>
""", unsafe_unsafe_allow_html=True) if hasattr(st, "markdown") else None

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
        color: #f8fafc;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# Load pipeline and dataset
pipe = joblib.load('pipeline.joblib')
df = load_breast_cancer(as_frame=True).frame
X = df.drop(columns='target')
y = df['target']
typical = df.groupby('target').mean()

st.title('🔬 Breast Tumour Risk Screener')
st.caption('Educational demo only. Not a medical device.')

with st.expander("ℹ️ About the Dataset & Metrics"):
    st.markdown("""
    - **Source**: Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
    - **Features**: 30 nuclear features computed from digitized images of Fine Needle Aspirates (FNA).
    - **Recall (Sensitivity)**: Measures the proportion of actual malignant cases correctly flagged.
    """)

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

threshold = st.slider("Decision Threshold (Safety Dial)", 0.10, 0.90, 0.30, 0.05)

# Main UI Grid
col_pred, col_shape = st.columns([1, 1])

with col_pred:
    if st.button('Predict'):
        p_malignant = 1 - pipe.predict_proba(pd.DataFrame([values]))[0][1]
        
        st.metric('Estimated chance of malignancy', f'{p_malignant:.0%}')
        
        if p_malignant > threshold:
            st.error(f'Flag for follow-up (Risk exceeds safety threshold of {threshold:.0%})')
        else:
            st.success('Low estimated risk')

with col_shape:
    st.write("**3D Cell Morphological Model (Interactive)**")
    
    # Extract morphological features
    r_val = values.get('mean radius', 14.0)
    concavity_val = values.get('mean concavity', 0.1)
    smoothness_val = values.get('mean smoothness', 0.1)
    
    # Generate 3D spherical mesh deformed by concavity and smoothness
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    
    # Irregularity displacement based on sliders
    deformation = (
        np.sin(4 * theta) * np.cos(3 * phi) * concavity_val * 12 +
        np.cos(8 * theta) * np.sin(6 * phi) * smoothness_val * 8
    )
    radius_3d = r_val + deformation
    
    # Spherical coordinates to Cartesian
    x_3d = radius_3d * np.sin(phi) * np.cos(theta)
    y_3d = radius_3d * np.sin(phi) * np.sin(theta)
    z_3d = radius_3d * np.cos(phi)
    
    # Pick color based on irregularity
    cell_color = 'Reds' if concavity_val > 0.15 else 'Greens'
    
    fig_3d = go.Figure(data=[go.Surface(
        x=x_3d, y=y_3d, z=z_3d,
        colorscale=cell_color,
        showscale=False,
        opacity=0.85
    )])
    
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=320
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

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