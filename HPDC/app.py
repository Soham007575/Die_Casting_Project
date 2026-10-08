import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# HPDC Die Casting - Streamlit Application
# Based on the supplied Model.ipynb
# ============================================================

st.set_page_config(
    page_title="HPDC AI Process Advisor",
    page_icon="🏭",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH =  BASE_DIR / "HPDC_1000_rows_11_columns_FIXED.csv"

FEATURES = [
    "melt_temp_C",
    "die_temp_C",
    "injection_speed_mps",
    "injection_pressure_bar",
    "intensification_pressure_bar",
    "filling_time_s",
    "cooling_time_s",
    "vacuum_pressure_bar",
    "alloy_type",
]

NUMERIC_FEATURES = [
    "melt_temp_C",
    "die_temp_C",
    "injection_speed_mps",
    "injection_pressure_bar",
    "intensification_pressure_bar",
    "filling_time_s",
    "cooling_time_s",
    "vacuum_pressure_bar",
]

CATEGORICAL_FEATURES = ["alloy_type"]


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Make sure app.py is in the Die-Casting project root."
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["quality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES].copy()
    y = df["quality"].map({"OK": 0, "DEFECT": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf_model),
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def predict_casting(model, values):
    input_data = pd.DataFrame([values])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    quality = "DEFECT" if prediction == 1 else "OK"

    if probability < 0.30:
        risk = "LOW"
    elif probability < 0.60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return quality, probability, risk


def generate_recommendation(model, df, current_process, n_candidates=5000):
    parameter_ranges = {
        parameter: (
            float(df[parameter].min()),
            float(df[parameter].max()),
        )
        for parameter in NUMERIC_FEATURES
    }

    np.random.seed(42)
    candidates = pd.DataFrame()

    for parameter, (minimum, maximum) in parameter_ranges.items():
        current_value = current_process[parameter]

        scale = (maximum - minimum) * 0.08
        values = np.random.normal(
            loc=current_value,
            scale=scale,
            size=n_candidates,
        )

        candidates[parameter] = np.clip(values, minimum, maximum)

    candidates["alloy_type"] = current_process["alloy_type"]

    candidates["defect_probability"] = model.predict_proba(
        candidates
    )[:, 1]

    best_candidates = candidates.sort_values(
        by="defect_probability",
        ascending=True,
    ).head(10).copy()

    for parameter in NUMERIC_FEATURES:
        minimum, maximum = parameter_ranges[parameter]
        parameter_range = maximum - minimum

        if parameter_range == 0:
            best_candidates[parameter + "_change"] = 0.0
        else:
            best_candidates[parameter + "_change"] = (
                abs(
                    best_candidates[parameter]
                    - current_process[parameter]
                )
                / parameter_range
            )

    change_columns = [
        parameter + "_change"
        for parameter in NUMERIC_FEATURES
    ]

    best_candidates["change_score"] = (
        best_candidates[change_columns].mean(axis=1)
    )

    best_candidates["optimization_score"] = (
        0.70 * best_candidates["defect_probability"]
        + 0.30 * best_candidates["change_score"]
    )

    best_candidates = best_candidates.sort_values(
        by="optimization_score",
        ascending=True,
    )

    return best_candidates.iloc[0], parameter_ranges


# ----------------------------
# Load data + model
# ----------------------------
try:
    df = load_data()
    model = train_model(df)
except Exception as e:
    st.error("Application setup failed.")
    st.exception(e)
    st.stop()


# ----------------------------
# Header
# ----------------------------
st.title("🏭 HPDC AI Process Advisor")
st.caption(
    "High Pressure Die Casting quality prediction and "
    "model-based process recommendation"
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select module",
    ["Prediction", "AI Process Advisor", "Dataset Insights"],
)

st.sidebar.divider()
st.sidebar.info(
    "Model: Random Forest\n\n"
    "Trees: 300\n\n"
    "Target: OK / DEFECT"
)


# ============================================================
# PREDICTION
# ============================================================
if page == "Prediction":

    st.header("Casting Quality Prediction")

    st.write(
        "Enter the current casting process parameters. "
        "The model predicts casting quality and defect probability."
    )

    alloy_options = sorted(
        df["alloy_type"].dropna().astype(str).unique().tolist()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        melt_temp = st.number_input(
            "Melt Temperature (°C)",
            min_value=float(df["melt_temp_C"].min()),
            max_value=float(df["melt_temp_C"].max()),
            value=680.0,
            step=1.0,
        )

        die_temp = st.number_input(
            "Die Temperature (°C)",
            min_value=float(df["die_temp_C"].min()),
            max_value=float(df["die_temp_C"].max()),
            value=190.0,
            step=1.0,
        )

        injection_speed = st.number_input(
            "Injection Speed (m/s)",
            min_value=float(df["injection_speed_mps"].min()),
            max_value=float(df["injection_speed_mps"].max()),
            value=2.50,
            step=0.01,
        )

    with col2:
        injection_pressure = st.number_input(
            "Injection Pressure (bar)",
            min_value=float(df["injection_pressure_bar"].min()),
            max_value=float(df["injection_pressure_bar"].max()),
            value=720.0,
            step=1.0,
        )

        intensification_pressure = st.number_input(
            "Intensification Pressure (bar)",
            min_value=float(df["intensification_pressure_bar"].min()),
            max_value=float(df["intensification_pressure_bar"].max()),
            value=620.0,
            step=1.0,
        )

        filling_time = st.number_input(
            "Filling Time (s)",
            min_value=float(df["filling_time_s"].min()),
            max_value=float(df["filling_time_s"].max()),
            value=0.045,
            step=0.001,
            format="%.3f",
        )

    with col3:
        cooling_time = st.number_input(
            "Cooling Time (s)",
            min_value=float(df["cooling_time_s"].min()),
            max_value=float(df["cooling_time_s"].max()),
            value=8.5,
            step=0.1,
        )

        vacuum_pressure = st.number_input(
            "Vacuum Pressure (bar)",
            min_value=float(df["vacuum_pressure_bar"].min()),
            max_value=float(df["vacuum_pressure_bar"].max()),
            value=-0.75,
            step=0.01,
        )

        alloy_type = st.selectbox(
            "Alloy Type",
            alloy_options,
            index=alloy_options.index("AlSi9Cu3")
            if "AlSi9Cu3" in alloy_options
            else 0,
        )

    values = {
        "melt_temp_C": melt_temp,
        "die_temp_C": die_temp,
        "injection_speed_mps": injection_speed,
        "injection_pressure_bar": injection_pressure,
        "intensification_pressure_bar": intensification_pressure,
        "filling_time_s": filling_time,
        "cooling_time_s": cooling_time,
        "vacuum_pressure_bar": vacuum_pressure,
        "alloy_type": alloy_type,
    }

    if st.button("🔮 Predict Casting Quality", type="primary"):
        quality, probability, risk = predict_casting(model, values)

        st.divider()

        r1, r2, r3 = st.columns(3)

        with r1:
            if quality == "OK":
                st.success("Predicted Quality: OK")
            else:
                st.error("Predicted Quality: DEFECT")

        with r2:
            st.metric(
                "Defect Probability",
                f"{probability * 100:.2f}%",
            )

        with r3:
            if risk == "LOW":
                st.success(f"Risk: {risk}")
            elif risk == "MEDIUM":
                st.warning(f"Risk: {risk}")
            else:
                st.error(f"Risk: {risk}")

        st.progress(probability)

        if quality == "DEFECT":
            st.warning(
                "The model predicts DEFECT for these process conditions. "
                "Review the process parameters before production."
            )
        else:
            st.info(
                "The model predicts OK for these process conditions."
            )


# ============================================================
# AI PROCESS ADVISOR
# ============================================================
elif page == "AI Process Advisor":

    st.header("🤖 AI Process Advisor")

    st.write(
        "Enter the current operating condition. The advisor generates "
        "5,000 candidate settings inside the observed dataset ranges "
        "and selects a model-based low-risk setting while considering "
        "the magnitude of parameter changes."
    )

    alloy_options = sorted(
        df["alloy_type"].dropna().astype(str).unique().tolist()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        current_melt = st.number_input(
            "Current Melt Temperature (°C)",
            min_value=float(df["melt_temp_C"].min()),
            max_value=float(df["melt_temp_C"].max()),
            value=680.0,
            step=1.0,
            key="advisor_melt",
        )
        current_die = st.number_input(
            "Current Die Temperature (°C)",
            min_value=float(df["die_temp_C"].min()),
            max_value=float(df["die_temp_C"].max()),
            value=190.0,
            step=1.0,
            key="advisor_die",
        )
        current_speed = st.number_input(
            "Current Injection Speed (m/s)",
            min_value=float(df["injection_speed_mps"].min()),
            max_value=float(df["injection_speed_mps"].max()),
            value=2.50,
            step=0.01,
            key="advisor_speed",
        )

    with c2:
        current_injection_pressure = st.number_input(
            "Current Injection Pressure (bar)",
            min_value=float(df["injection_pressure_bar"].min()),
            max_value=float(df["injection_pressure_bar"].max()),
            value=720.0,
            step=1.0,
            key="advisor_inj_pressure",
        )
        current_intensification = st.number_input(
            "Current Intensification Pressure (bar)",
            min_value=float(df["intensification_pressure_bar"].min()),
            max_value=float(df["intensification_pressure_bar"].max()),
            value=620.0,
            step=1.0,
            key="advisor_int_pressure",
        )
        current_filling = st.number_input(
            "Current Filling Time (s)",
            min_value=float(df["filling_time_s"].min()),
            max_value=float(df["filling_time_s"].max()),
            value=0.045,
            step=0.001,
            format="%.3f",
            key="advisor_filling",
        )

    with c3:
        current_cooling = st.number_input(
            "Current Cooling Time (s)",
            min_value=float(df["cooling_time_s"].min()),
            max_value=float(df["cooling_time_s"].max()),
            value=8.5,
            step=0.1,
            key="advisor_cooling",
        )
        current_vacuum = st.number_input(
            "Current Vacuum Pressure (bar)",
            min_value=float(df["vacuum_pressure_bar"].min()),
            max_value=float(df["vacuum_pressure_bar"].max()),
            value=-0.75,
            step=0.01,
            key="advisor_vacuum",
        )
        current_alloy = st.selectbox(
            "Current Alloy Type",
            alloy_options,
            index=alloy_options.index("AlSi9Cu3")
            if "AlSi9Cu3" in alloy_options
            else 0,
            key="advisor_alloy",
        )

    current_process = {
        "melt_temp_C": current_melt,
        "die_temp_C": current_die,
        "injection_speed_mps": current_speed,
        "injection_pressure_bar": current_injection_pressure,
        "intensification_pressure_bar": current_intensification,
        "filling_time_s": current_filling,
        "cooling_time_s": current_cooling,
        "vacuum_pressure_bar": current_vacuum,
        "alloy_type": current_alloy,
    }

    if st.button("⚙️ Generate AI Recommendation", type="primary"):
        current_quality, current_probability, current_risk = (
            predict_casting(model, current_process)
        )

        with st.spinner("Evaluating 5,000 candidate process settings..."):
            best, parameter_ranges = generate_recommendation(
                model,
                df,
                current_process,
                n_candidates=5000,
            )

        st.divider()

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Current Defect Risk",
                f"{current_probability * 100:.2f}%",
            )

        with b:
            st.metric(
                "Recommended Defect Risk",
                f"{best['defect_probability'] * 100:.2f}%",
            )

        with c:
            risk_reduction = (
                current_probability
                - float(best["defect_probability"])
            ) * 100

            st.metric(
                "Predicted Risk Reduction",
                f"{risk_reduction:.2f} percentage points",
            )

        st.subheader("Recommended Process Parameters")

        comparison = pd.DataFrame({
            "Parameter": NUMERIC_FEATURES,
            "Current": [
                current_process[p]
                for p in NUMERIC_FEATURES
            ],
            "Recommended": [
                float(best[p])
                for p in NUMERIC_FEATURES
            ],
        })

        comparison["Change"] = (
            comparison["Recommended"]
            - comparison["Current"]
        )

        st.dataframe(
            comparison.style.format({
                "Current": "{:.4f}",
                "Recommended": "{:.4f}",
                "Change": "{:+.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Recommendation is model-based. Validate any process change "
            "before applying it to an actual production machine."
        )


# ============================================================
# DATASET INSIGHTS
# ============================================================
else:

    st.header("📊 Dataset Insights")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Rows", f"{len(df):,}")

    with m2:
        st.metric("Columns", f"{df.shape[1]}")

    with m3:
        defect_rate = (
            (df["quality"] == "DEFECT").mean() * 100
        )
        st.metric("Observed Defect Rate", f"{defect_rate:.2f}%")

    st.subheader("Quality Distribution")
    quality_counts = df["quality"].value_counts()
    st.bar_chart(quality_counts)

    st.subheader("Defect Type Distribution")
    if "defect_type" in df.columns:
        st.bar_chart(df["defect_type"].value_counts())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Process Parameter Ranges")

    ranges = pd.DataFrame({
        "Parameter": NUMERIC_FEATURES,
        "Minimum": [df[c].min() for c in NUMERIC_FEATURES],
        "Maximum": [df[c].max() for c in NUMERIC_FEATURES],
        "Mean": [df[c].mean() for c in NUMERIC_FEATURES],
    })

    st.dataframe(
        ranges.style.format({
            "Minimum": "{:.4f}",
            "Maximum": "{:.4f}",
            "Mean": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "HPDC AI Process Advisor • Model logic based on the supplied Model.ipynb"
)
