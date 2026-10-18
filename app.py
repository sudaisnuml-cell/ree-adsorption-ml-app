
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="REE Adsorption Predictor", layout="wide")
DATA_FILE = "Sudais_REE_AI_Model_Dataset_Clear_Inputs_Output.xlsx"

@st.cache_resource
def train_model():
    df = pd.read_excel(DATA_FILE, sheet_name="ML_Ready", header=3)
    cols = [
        "Metal_INPUT","System_INPUT","Experiment_INPUT",
        "Contact_Time_min_INPUT","C0_mg_L_INPUT",
        "Adsorbent_Dose_g_L_INPUT","qe_mg_g_OUTPUT"
    ]
    df = df[cols].dropna().copy()
    X = df[cols[:-1]]
    y = df["qe_mg_g_OUTPUT"]

    cat = ["Metal_INPUT","System_INPUT","Experiment_INPUT"]
    num = ["Contact_Time_min_INPUT","C0_mg_L_INPUT","Adsorbent_Dose_g_L_INPUT"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ])

    model = Pipeline([
        ("preprocessor", pre),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return model, df, y_test, y_pred, metrics

model, df, y_test, y_pred, metrics = train_model()

st.title("Rare-Earth Adsorption Capacity Predictor")
st.warning(
    "All inputs are editable. If you choose values or combinations not represented "
    "in the training data, the model can still give a number, but that prediction is less reliable."
)

left, right = st.columns(2)

with left:
    st.subheader("Model Inputs — All Editable")
    metal = st.selectbox("Metal", ["Pr", "Tm"])
    system = st.selectbox("System", ["Single", "Binary"])
    experiment = st.selectbox("Experiment Type", ["Uptake kinetics", "Sorption isotherm"])

    contact_time = st.number_input("Contact Time (min)", min_value=0.0, value=20.0, step=1.0)
    c0 = st.number_input("Initial Concentration, C0 (mg/L)", min_value=0.0, value=50.0, step=10.0)
    dose = st.number_input("Adsorbent Dose (g/L)", min_value=0.01, value=0.25, step=0.05)

    combo_exists = (
        (df["Metal_INPUT"] == metal) &
        (df["System_INPUT"] == system) &
        (df["Experiment_INPUT"] == experiment)
    ).any()

    if not combo_exists:
        st.error("This categorical combination was not present in the training dataset.")

    if st.button("Predict adsorption capacity", type="primary"):
        row = pd.DataFrame({
            "Metal_INPUT":[metal],
            "System_INPUT":[system],
            "Experiment_INPUT":[experiment],
            "Contact_Time_min_INPUT":[contact_time],
            "C0_mg_L_INPUT":[c0],
            "Adsorbent_Dose_g_L_INPUT":[dose]
        })
        pred = model.predict(row)[0]
        st.success(f"Predicted adsorption capacity = {pred:.3f} mg/g")

with right:
    st.subheader("Model Performance")
    a,b,c = st.columns(3)
    a.metric("R²", f"{metrics['R2']:.4f}")
    b.metric("MAE", f"{metrics['MAE']:.4f} mg/g")
    c.metric("RMSE", f"{metrics['RMSE']:.4f} mg/g")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn,mx],[mn,mx], linestyle="--")
    ax.set_xlabel("Actual adsorption capacity (mg/g)")
    ax.set_ylabel("Predicted adsorption capacity (mg/g)")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
