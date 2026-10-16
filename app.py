
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CreditWise AI | Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "credit_model.joblib"

ENGINEERED_FEATURES = {
    "income_missing",
    "total_late",
    "ever_90_late",
    "log_income",
    "monthly_debt",
    "income_per_person",
    "high_utilization",
    "age_group",
}

# Original Give Me Some Credit features
FIELD_CONFIG = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Revolving Credit Utilization",
        "help": "Fraction of available unsecured credit being used.",
        "min": 0.0,
        "max": 100.0,
        "value": 0.30,
        "step": 0.01,
        "format": "%.2f",
    },
    "age": {
        "label": "Applicant Age",
        "help": "Age in years.",
        "min": 18,
        "max": 100,
        "value": 35,
        "step": 1,
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "30–59 Days Past Due",
        "help": "Number of times payments were 30–59 days late.",
        "min": 0,
        "max": 100,
        "value": 0,
        "step": 1,
    },
    "DebtRatio": {
        "label": "Debt Ratio",
        "help": "Debt-to-income-related ratio in the dataset.",
        "min": 0.0,
        "max": 1000.0,
        "value": 0.35,
        "step": 0.01,
        "format": "%.2f",
    },
    "MonthlyIncome": {
        "label": "Monthly Income (₹)",
        "help": "Enter monthly income in the same units used during training.",
        "min": 0.0,
        "max": 10000000.0,
        "value": 50000.0,
        "step": 1000.0,
        "format": "%.0f",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open Credit Lines and Loans",
        "help": "Total number of open credit lines and loans.",
        "min": 0,
        "max": 100,
        "value": 5,
        "step": 1,
    },
    "NumberOfTimes90DaysLate": {
        "label": "90+ Days Past Due",
        "help": "Number of times payments were 90 days or more late.",
        "min": 0,
        "max": 100,
        "value": 0,
        "step": 1,
    },
    "NumberRealEstateLoansOrLines": {
        "label": "Real Estate Loans or Lines",
        "help": "Number of real estate loans or credit lines.",
        "min": 0,
        "max": 100,
        "value": 1,
        "step": 1,
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "60–89 Days Past Due",
        "help": "Number of times payments were 60–89 days late.",
        "min": 0,
        "max": 100,
        "value": 0,
        "step": 1,
    },
    "NumberOfDependents": {
        "label": "Number of Dependents",
        "help": "Number of dependents in the household.",
        "min": 0,
        "max": 20,
        "value": 0,
        "step": 1,
    },
}

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #101b30 100%);
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: #0a1020;
        border-right: 1px solid #243247;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
    }
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 16px;
    }
    .hero {
        padding: 24px;
        border: 1px solid #334155;
        border-radius: 18px;
        background: linear-gradient(110deg, #172554, #172033);
        margin-bottom: 20px;
    }
    .hero h1 { margin-bottom: 6px; }
    .muted { color: #a8b5c9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL ARTIFACT
# --------------------------------------------------
@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Place credit_model.joblib in the same folder as app.py."
    )
    st.stop()

try:

    artifact = load_artifact(str(MODEL_PATH))
    model = artifact["model"]
    features = list(artifact["features"])
    threshold = float(artifact.get("threshold", 0.5))

    # Compatibility repair for older fitted SimpleImputer objects.
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer


    def repair_imputers(obj):
        if isinstance(obj, SimpleImputer):
            if not hasattr(obj, "_fill_dtype") and hasattr(obj, "statistics_"):
                obj._fill_dtype = obj.statistics_.dtype

        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                repair_imputers(step)

        elif isinstance(obj, ColumnTransformer):
            for _, transformer, _ in obj.transformers_:
                if transformer not in ("drop", "passthrough"):
                    repair_imputers(transformer)

        elif hasattr(obj, "estimator") and obj.estimator is not None:
            repair_imputers(obj.estimator)

        elif hasattr(obj, "transformer_list"):
            for _, transformer in obj.transformer_list:
                repair_imputers(transformer)


    repair_imputers(model)

except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

# --------------------------------------------------
# FEATURE VALIDATION
# --------------------------------------------------
input_features = [
    feature for feature in features
    if feature not in ENGINEERED_FEATURES
]

unsupported_features = [
    feature for feature in input_features
    if feature not in FIELD_CONFIG
]

if unsupported_features:
    st.error(
        "Your saved model contains original input features that this "
        f"UI does not support yet: {unsupported_features}. "
        "Update FIELD_CONFIG to match your training notebook."
    )
    st.stop()

required_engineered = ENGINEERED_FEATURES.intersection(features)
expected_engineered = ENGINEERED_FEATURES.intersection(features)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💳 CreditWise AI</h1>
        <p class="muted">
            Interactive credit-risk analysis powered by your trained
            machine-learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Dashboard")
    st.caption("Credit risk assessment")
    st.divider()

    st.markdown("### Model information")
    st.write("**Algorithm:** Your saved model")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    st.write(f"**Model inputs:** {len(features)} features")

    st.divider()
    st.markdown("### How it works")
    st.write("1. Enter applicant information.")
    st.write("2. Generate engineered features.")
    st.write("3. Calculate the model score.")
    st.write("4. Review the prediction and risk indicators.")

    st.info(
        "Educational project only. This tool is not a validated "
        "lending decision system."
    )

# --------------------------------------------------
# APPLICANT INPUT FORM
# --------------------------------------------------
st.subheader("👤 Applicant profile")
st.write(
    "Enter the applicant's details below. Engineered features "
    "are calculated automatically."
)

with st.form("applicant_form", clear_on_submit=False):
    st.markdown("#### Personal and financial details")
    col1, col2 = st.columns(2)

    values = {}

    # Keep the raw income value separate so the app can represent
    # an unavailable income correctly.
    with col1:
        income_missing = st.checkbox(
            "Monthly income is unavailable",
            value=False,
            help="Select this if income is genuinely missing.",
        )

    for index, feature in enumerate(input_features):
        config = FIELD_CONFIG[feature]
        target_col = col1 if index % 2 == 0 else col2

        with target_col:
            if feature == "MonthlyIncome" and income_missing:
                values[feature] = np.nan
                st.text_input(
                    config["label"],
                    value="Not provided",
                    disabled=True,
                    help=config["help"],
                )
            else:
                kwargs = {
                    "label": config["label"],
                    "min_value": config["min"],
                    "max_value": config["max"],
                    "value": config["value"],
                    "step": config["step"],
                    "help": config["help"],
                    "key": f"input_{feature}",
                }
                if "format" in config:
                    kwargs["format"] = config["format"]

                values[feature] = st.number_input(**kwargs)

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Analyze Credit Risk",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def create_model_row(raw_values, model_features):
    """Create a single prediction row in the trained feature order."""

    row = pd.DataFrame(
        [{name: raw_values[name] for name in input_features}],
        columns=input_features,
    )

    # Preserve the missing-income indicator before model imputation.
    row["income_missing"] = row["MonthlyIncome"].isna().astype(int)

    row["total_late"] = (
        row["NumberOfTime30-59DaysPastDueNotWorse"]
        + row["NumberOfTime60-89DaysPastDueNotWorse"]
        + row["NumberOfTimes90DaysLate"]
    )

    row["ever_90_late"] = (
        row["NumberOfTimes90DaysLate"] > 0
    ).astype(int)

    row["log_income"] = np.log1p(
        row["MonthlyIncome"].clip(lower=0)
    )

    row["monthly_debt"] = (
        row["DebtRatio"] * row["MonthlyIncome"]
    )

    row["income_per_person"] = (
        row["MonthlyIncome"]
        / (row["NumberOfDependents"] + 1)
    )

    row["high_utilization"] = (
        row["RevolvingUtilizationOfUnsecuredLines"] > 0.8
    ).astype(int)

    row["age_group"] = pd.cut(
        row["age"],
        bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
        labels=False,
        right=True,
    ).astype(int)

    # Ensure all expected columns exist and match training order.
    missing_columns = [name for name in model_features if name not in row.columns]
    if missing_columns:
        raise ValueError(
            f"Feature engineering did not create: {missing_columns}"
        )

    return row.reindex(columns=model_features)


# --------------------------------------------------
# RISK GAUGE
# --------------------------------------------------
def create_gauge(probability):
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 38}},
            title={"text": "Model-estimated risk score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 25]},
                    {"range": [25, 50]},
                    {"range": [50, 75]},
                    {"range": [75, 100]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )

    figure.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
    )
    return figure


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
if submitted:
    try:
        # Ensure the app's input dictionary is complete.
        missing_inputs = [
            name for name in input_features if name not in values
        ]
        if missing_inputs:
            st.error(f"Missing applicant inputs: {missing_inputs}")
            st.stop()

        # Basic validation for impossible negative values.
        nonnegative_features = [
            name for name in input_features
            if name != "age"
        ]
        invalid_values = [
            name for name in nonnegative_features
            if pd.notna(values[name]) and values[name] < 0
        ]

        if invalid_values:
            st.error(
                "These fields cannot be negative: "
                + ", ".join(invalid_values)
            )
            st.stop()

        if values["age"] < 18 or values["age"] > 100:
            st.error("Please enter an age between 18 and 100.")
            st.stop()

        row = create_model_row(values, features)

        # The saved pipeline should perform its own fitted imputation
        # and preprocessing.
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(probability >= threshold)

        st.divider()
        st.subheader("📊 Credit-risk results")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Estimated risk score",
                f"{probability:.2%}",
            )

        with metric2:
            st.metric(
                "Decision threshold",
                f"{threshold:.2%}",
            )

        with metric3:
            st.metric(
                "Model classification",
                "Higher risk flag" if prediction else "Lower risk flag",
            )

        result_col, gauge_col = st.columns([1, 1.2])

        with result_col:
            if prediction:
                st.error("### ⚠️ Higher-risk flag")
                st.write(
                    "The model score is at or above the configured "
                    "classification threshold."
                )
            else:
                st.success("### ✅ Lower-risk flag")
                st.write(
                    "The model score is below the configured "
                    "classification threshold."
                )

            st.caption(
                "This is a model classification, not a loan approval "
                "or rejection recommendation."
            )

        with gauge_col:
            st.plotly_chart(
                create_gauge(probability),
                use_container_width=True,
            )

        st.subheader("🔎 Applicant indicators")

        indicator_col1, indicator_col2 = st.columns(2)

        late_total = int(row["total_late"].iloc[0])
        utilization = float(
            values["RevolvingUtilizationOfUnsecuredLines"]
        )
        age = int(values["age"])

        with indicator_col1:
            st.markdown("**Payment history**")
            st.write(f"Recorded late-payment events: **{late_total}**")

            if late_total > 0:
                st.warning(
                    "The entered history includes late-payment events."
                )
            else:
                st.info(
                    "No late-payment events were entered in the supplied fields."
                )

        with indicator_col2:
            st.markdown("**Credit utilization**")
            st.write(f"Entered utilization: **{utilization:.2f}**")

            if utilization > 0.8:
                st.warning(
                    "Utilization is above the app's 0.80 indicator threshold."
                )
            else:
                st.info(
                    "Utilization is at or below the app's 0.80 indicator threshold."
                )

        st.subheader("🧾 Submitted input summary")
        display_values = {
            FIELD_CONFIG[name]["label"]: (
                "Not provided"
                if pd.isna(values[name])
                else values[name]
            )
            for name in input_features
        }
        st.dataframe(
            pd.DataFrame(
                list(display_values.items()),
                columns=["Input", "Value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View model-ready features"):
            st.caption(
                "These values are the features passed to the saved model. "
                "Missing values may be imputed inside the model pipeline."
            )
            st.dataframe(row, use_container_width=True)

        # Export a small prediction report.
        report = pd.DataFrame([{
            "estimated_probability": probability,
            "threshold": threshold,
            "classification": (
                "Higher-risk flag" if prediction else "Lower-risk flag"
            ),
            **display_values,
        }])

        st.download_button(
            "⬇️ Download prediction report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="credit_risk_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that your saved model, feature names, and feature "
            "engineering formulas match the training notebook."
        )
else:
    st.info(
        "Complete the applicant form and click **Analyze Credit Risk** "
        "to view the model's result."
    )

st.divider()
st.caption(
    "CreditWise AI · Machine Learning internship project · "
    "For educational demonstration only"
)
