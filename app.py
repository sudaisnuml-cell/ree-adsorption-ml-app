
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

st.set_page_config(
    page_title="REE Adsorption Predictor",
    page_icon="🧪",
    layout="wide"
)

DATA_FILE = "Sudais_REE_AI_Model_Dataset_Clear_Inputs_Output.xlsx"

@st.cache_resource
def train_model():
    df = pd.read_excel(DATA_FILE, sheet_name="ML_Ready", header=3)

    required_cols = [
        "Metal_INPUT",
        "System_INPUT",
        "Experiment_INPUT",
        "Contact_Time_min_INPUT",
        "C0_mg_L_INPUT",
        "Adsorbent_Dose_g_L_INPUT",
        "qe_mg_g_OUTPUT"
    ]
    df = df[required_cols].dropna().copy()

    X = df[
        [
            "Metal_INPUT",
            "System_INPUT",
            "Experiment_INPUT",
            "Contact_Time_min_INPUT",
            "C0_mg_L_INPUT",
            "Adsorbent_Dose_g_L_INPUT"
        ]
    ]
    y = df["qe_mg_g_OUTPUT"]

    categorical_features = [
        "Metal_INPUT",
        "System_INPUT",
        "Experiment_INPUT"
    ]
    numeric_features = [
        "Contact_Time_min_INPUT",
        "C0_mg_L_INPUT",
        "Adsorbent_Dose_g_L_INPUT"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42
                )
            )
        ]
    )

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

    return model, df, X_test, y_test, y_pred, metrics


model, df, X_test, y_test, y_pred, metrics = train_model()

st.title("Rare-Earth Adsorption Capacity Predictor")
st.caption(
    "Random Forest Regression model trained on literature-derived P2-APEI adsorption data "
    "for Pr(III) and Tm(III)."
)

st.info(
    "Use the controls below and click Predict. "
    "The app keeps scientifically valid experiment/system combinations."
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Model Inputs")

    metal = st.selectbox(
        "Metal",
        options=["Pr", "Tm"],
        help="Pr = Praseodymium, Tm = Thulium"
    )

    experiment = st.selectbox(
        "Experiment Type",
        options=["Uptake kinetics", "Sorption isotherm"]
    )

    if experiment == "Sorption isotherm":
        system = "Single"
        st.text_input("System", value="Single", disabled=True)
        contact_time = 2880.0
        st.number_input(
            "Contact Time (min)",
            value=2880.0,
            disabled=True,
            help="48 h = 2880 min in the source isotherm experiment."
        )
        c0 = st.number_input(
            "Initial Concentration, C0 (mg/L)",
            min_value=10.0,
            max_value=500.0,
            value=50.0,
            step=10.0
        )
        dose = 0.5
        st.number_input(
            "Adsorbent Dose (g/L)",
            value=0.5,
            disabled=True
        )
    else:
        system = st.selectbox(
            "System",
            options=["Single", "Binary"],
            help="Single = one REE; Binary = Pr and Tm together."
        )
        contact_time = st.number_input(
            "Contact Time (min)",
            min_value=0.0,
            max_value=60.0,
            value=20.0,
            step=1.0
        )
        c0 = 50.0
        st.number_input(
            "Initial Concentration, C0 (mg/L)",
            value=50.0,
            disabled=True
        )
        dose = 0.25
        st.number_input(
            "Adsorbent Dose (g/L)",
            value=0.25,
            disabled=True
        )

    if st.button("Predict adsorption capacity", type="primary"):
        new_input = pd.DataFrame(
            {
                "Metal_INPUT": [metal],
                "System_INPUT": [system],
                "Experiment_INPUT": [experiment],
                "Contact_Time_min_INPUT": [contact_time],
                "C0_mg_L_INPUT": [c0],
                "Adsorbent_Dose_g_L_INPUT": [dose]
            }
        )

        prediction = model.predict(new_input)[0]

        st.success(
            f"Predicted adsorption capacity = {prediction:.3f} mg/g"
        )

        st.caption(
            "Prediction is most reliable within the experimental range represented by the training dataset."
        )

with right:
    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R2']:.4f}")
    m2.metric("MAE", f"{metrics['MAE']:.4f} mg/g")
    m3.metric("RMSE", f"{metrics['RMSE']:.4f} mg/g")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Actual adsorption capacity (mg/g)")
    ax.set_ylabel("Predicted adsorption capacity (mg/g)")
    ax.set_title("Actual vs Predicted")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.divider()
st.subheader("Feature Importance")

ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
categorical_features = ["Metal_INPUT", "System_INPUT", "Experiment_INPUT"]
numeric_features = [
    "Contact_Time_min_INPUT",
    "C0_mg_L_INPUT",
    "Adsorbent_Dose_g_L_INPUT"
]

encoded_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = encoded_names + numeric_features
importances = model.named_steps["regressor"].feature_importances_

fi = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values("Importance", ascending=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(fi["Feature"], fi["Importance"])
ax2.set_xlabel("Importance")
ax2.set_title("Random Forest Feature Importance")
plt.tight_layout()
st.pyplot(fig2)

with st.expander("Project / Viva Notes"):
    st.markdown(
        """
- **Machine-learning type:** Supervised learning  
- **Task:** Regression  
- **Algorithm:** Random Forest Regression  
- **Inputs:** Metal, system, experiment type, contact time, initial concentration, adsorbent dose  
- **Output:** Adsorption capacity in mg/g  
- **Validation:** R², MAE, RMSE  
- **Important limitation:** This model is trained on the literature P2-APEI dataset and should mainly be used within the represented experimental domain.
        """
    )
