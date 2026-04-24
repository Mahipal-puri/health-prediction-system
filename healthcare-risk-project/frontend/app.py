"""
Smart Healthcare Risk Prediction System — Indian Edition
Interactive Streamlit Frontend

Predicts patient health risk using MLP Neural Network.
All costs displayed in Indian Rupees (₹). Regions mapped to Indian zones.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Constants ────────────────────────────────────────────────────────
USD_TO_INR = 83.0  # Conversion rate

# Indian region mapping (model internally uses these keys)
INDIAN_REGIONS = {
    "North India": "Northeast",
    "South India": "Southeast",
    "East India": "Southwest",
    "West India": "Northwest",
}
REGION_LIST = list(INDIAN_REGIONS.keys())

# Reverse map for display
REGION_DISPLAY = {v.lower(): k for k, v in INDIAN_REGIONS.items()}


def fmt_inr(amount_usd):
    """Format USD amount as Indian Rupees with comma style."""
    inr = amount_usd * USD_TO_INR
    return f"₹{inr:,.0f}"


# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇳 Smart Healthcare Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Load Models ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, "..", "model")
    mlp = joblib.load(os.path.join(model_dir, "mlp_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    lr = joblib.load(os.path.join(model_dir, "lr_model.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
    return mlp, scaler, lr, feature_cols


@st.cache_data
def load_dataset():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "..", "dataset", "insurance.csv")
    df = pd.read_csv(csv_path)
    df["risk"] = (df["charges"] > 15000).astype(int)
    df["charges_inr"] = (df["charges"] * USD_TO_INR).round(2)
    df["region_india"] = df["region"].map({
        "northeast": "North India",
        "southeast": "South India",
        "southwest": "East India",
        "northwest": "West India",
    })
    return df


try:
    mlp_model, scaler, lr_model, feature_cols = load_models()
    models_loaded = True
except Exception:
    models_loaded = False

df = load_dataset()

RISK_THRESHOLD_INR = (15000 * USD_TO_INR) / 10  # ₹1,24,500

# ─── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Risk Prediction",
        "🔄 Compare Scenarios",
        " Data Explorer",
        "📈 Model Performance",
        "🗺️ Regional Analysis",
    ],
)

# ─── Helpers ─────────────────────────────────────────────────────────

def prepare_input(age, bmi, children, sex, smoker, region_indian):
    """Convert user inputs to model-ready feature array."""
    model_region = INDIAN_REGIONS[region_indian]
    sex_encoded = 1 if sex == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0
    region_ne = 1 if model_region == "Northeast" else 0
    region_nw = 1 if model_region == "Northwest" else 0
    region_se = 1 if model_region == "Southeast" else 0
    region_sw = 1 if model_region == "Southwest" else 0

    features = np.array([[age, bmi, children, sex_encoded, smoker_encoded,
                           region_ne, region_nw, region_se, region_sw]])
    return features


def predict_single(age, bmi, children, sex, smoker, region_indian):
    """Run both models and return results dict with INR values."""
    features = prepare_input(age, bmi, children, sex, smoker, region_indian)
    features_scaled = scaler.transform(features)
    risk_pred = mlp_model.predict(features_scaled)[0]
    risk_proba = mlp_model.predict_proba(features_scaled)[0]
    cost_usd = max(lr_model.predict(features)[0], 0)
    cost_inr = (cost_usd * USD_TO_INR) / 10
    return {
        "risk": risk_pred,
        "risk_label": "HIGH RISK ⚠️" if risk_pred == 1 else "LOW RISK ✅",
        "low_prob": risk_proba[0],
        "high_prob": risk_proba[1],
        "confidence": max(risk_proba[0], risk_proba[1]),
        "cost_usd": cost_usd,
        "cost_inr": cost_inr,
    }


def bmi_category_india(bmi):
    """BMI classification as per Indian / Asian standards (WHO Asia-Pacific)."""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 23.0:
        return "Normal", "🟢"
    elif bmi < 25.0:
        return "Overweight", "🟡"
    elif bmi < 30.0:
        return "Obese Class I", "🟠"
    else:
        return "Obese Class II", "🔴"


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════
if page == "🔮 Risk Prediction":
    st.title("🏥 Smart Healthcare Risk Predictor 🇮🇳")
    st.markdown("Enter patient details and get an instant **risk prediction** with estimated medical cost in **₹ (Indian Rupees)**.")
    st.markdown("---")

    if not models_loaded:
        st.error("⚠️ Models not found! Run `python model/train_model.py` first.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Patient Details")
        age = st.slider("Age (years)", 18, 65, 30)
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 55.0, 23.0, 0.1,
                         help="Indian standard: Normal BMI is 18.5–23.0")
        children = st.number_input("Number of Children / Dependents", 0, 10, 0)

    with col2:
        st.subheader("📋 Additional Info")
        sex = st.selectbox("Gender", ["Male", "Female"])
        smoker = st.selectbox("Tobacco / Smoking", ["No", "Yes"])
        region = st.selectbox("Region (India)", REGION_LIST)

    # Live BMI status
    bmi_cat, bmi_icon = bmi_category_india(bmi)
    st.info(f"{bmi_icon} **BMI Category (Asian/Indian Standard):** {bmi_cat} — BMI {bmi:.1f}")

    st.markdown("---")

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        res = predict_single(age, bmi, children, sex, smoker, region)

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            if res["risk"] == 1:
                st.error("⚠️ **HIGH RISK**")
            else:
                st.success("✅ **LOW RISK**")
        with r2:
            st.metric("Estimated Medical Cost", f"₹{res['cost_inr']:,.0f}")
        with r3:
            st.metric("Confidence", f"{res['confidence'] * 100:.1f}%")

        # Cost breakdown
        st.markdown("#### 💰 Cost Details (₹)")
        cost_c1, cost_c2 = st.columns(2)
        with cost_c1:
            st.write(f"**Annual Estimated Cost:** ₹{res['cost_inr']:,.0f}")
            st.write(f"**Monthly Estimate:** ₹{res['cost_inr']/12:,.0f}")
        with cost_c2:
            st.write(f"**Risk Threshold:** ₹{RISK_THRESHOLD_INR:,.0f}")
            above_below = "ABOVE" if res['cost_inr'] > RISK_THRESHOLD_INR else "BELOW"
            st.write(f"**Your cost is:** {above_below} the threshold")

        # Probability breakdown
        st.subheader("Risk Probability Breakdown")
        prob_c1, prob_c2 = st.columns(2)
        with prob_c1:
            st.progress(res["low_prob"], text=f"Low Risk: {res['low_prob']*100:.2f}%")
        with prob_c2:
            st.progress(res["high_prob"], text=f"High Risk: {res['high_prob']*100:.2f}%")

        # Health recommendations (Indian context)
        st.subheader("💡 Health Recommendations")
        tips = []
        if smoker == "Yes":
            tips.append("🚭 **Quit Tobacco** — Tobacco use (smoking/gutkha/paan) is the #1 factor driving high medical costs. Consult a de-addiction clinic or call National Tobacco Quitline: **1800-11-2356**.")
        if bmi >= 25.0:
            tips.append("🏃 **Manage Weight** — Your BMI indicates obesity by Indian standards. Include yoga, walking, and a balanced Indian diet (roti, sabzi, dal).")
        elif bmi >= 23.0:
            tips.append("🥗 **Watch Your Weight** — BMI is in the overweight range (Indian standard: 23–25). Reduce oil/ghee intake and increase physical activity.")
        if age > 45:
            tips.append("🩺 **Regular Health Checkups** — Get annual checkups including blood sugar, BP, cholesterol, and ECG. Use Ayushman Bharat if eligible.")
        if children >= 3:
            tips.append("👨‍👩‍👧‍👦 **Family Health Insurance** — With multiple dependents, consider a family floater policy for comprehensive coverage.")
        if not tips:
            tips.append("👍 **Healthy Lifestyle!** — Your parameters look good. Continue with regular exercise, balanced diet, and annual checkups.")
        for tip in tips:
            st.markdown(tip)

        # Patient summary
        st.subheader("📋 Patient Summary")
        st.table(pd.DataFrame({
            "Parameter": ["Age", "BMI", "BMI Category (Indian)", "Children", "Gender", "Tobacco Use", "Region"],
            "Value": [f"{age} years", f"{bmi:.1f}", bmi_cat, children, sex, smoker, region],
        }))


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPARE SCENARIOS
# ══════════════════════════════════════════════════════════════════════
elif page == "🔄 Compare Scenarios":
    st.title("🔄 What-If Scenario Comparison 🇮🇳")
    st.markdown("Compare **two patient profiles** side-by-side — e.g. *'What if I quit smoking?'*")
    st.markdown("---")

    if not models_loaded:
        st.error("⚠️ Models not found!"); st.stop()

    col_a, col_sep, col_b = st.columns([5, 1, 5])

    with col_a:
        st.subheader("👤 Scenario A")
        age_a = st.number_input("Age", 18, 65, 30, key="a_age")
        bmi_a = st.number_input("BMI", 15.0, 55.0, 23.0, 0.1, key="a_bmi")
        children_a = st.number_input("Children", 0, 10, 0, key="a_child")
        sex_a = st.selectbox("Gender", ["Male", "Female"], key="a_sex")
        smoker_a = st.selectbox("Tobacco", ["No", "Yes"], key="a_smoke")
        region_a = st.selectbox("Region", REGION_LIST, key="a_reg")

    with col_sep:
        st.markdown("<div style='text-align:center; padding-top:200px; font-size:2rem;'>⚡</div>",
                    unsafe_allow_html=True)

    with col_b:
        st.subheader("👤 Scenario B")
        age_b = st.number_input("Age", 18, 65, 30, key="b_age")
        bmi_b = st.number_input("BMI", 15.0, 55.0, 23.0, 0.1, key="b_bmi")
        children_b = st.number_input("Children", 0, 10, 0, key="b_child")
        sex_b = st.selectbox("Gender", ["Male", "Female"], key="b_sex")
        smoker_b = st.selectbox("Tobacco", ["No", "Yes"], key="b_smoke")
        region_b = st.selectbox("Region", REGION_LIST, key="b_reg")

    st.markdown("---")

    if st.button("🔄 Compare Both Scenarios", type="primary", use_container_width=True):
        res_a = predict_single(age_a, bmi_a, children_a, sex_a, smoker_a, region_a)
        res_b = predict_single(age_b, bmi_b, children_b, sex_b, smoker_b, region_b)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Scenario A")
            if res_a["risk"] == 1:
                st.error("⚠️ **HIGH RISK**")
            else:
                st.success("✅ **LOW RISK**")
            st.metric("Estimated Cost", f"₹{res_a['cost_inr']:,.0f}")
            st.metric("High-Risk Probability", f"{res_a['high_prob']*100:.2f}%")

        with c2:
            st.markdown("### Scenario B")
            if res_b["risk"] == 1:
                st.error("⚠️ **HIGH RISK**")
            else:
                st.success("✅ **LOW RISK**")
            st.metric("Estimated Cost", f"₹{res_b['cost_inr']:,.0f}")
            st.metric("High-Risk Probability", f"{res_b['high_prob']*100:.2f}%")

        diff_inr = res_b["cost_inr"] - res_a["cost_inr"]
        st.markdown("---")
        if abs(diff_inr) > 100:
            direction = "more" if diff_inr > 0 else "less"
            st.info(f"💰 Scenario B costs **₹{abs(diff_inr):,.0f} {direction}** than Scenario A.")
        else:
            st.info("💰 Both scenarios have nearly the same estimated cost.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Interactive Dataset Explorer 🇮🇳")
    st.markdown("Filter and visualize the dataset. All amounts shown in **₹ (Indian Rupees)**.")
    st.markdown("---")

    # Sidebar filters
    st.sidebar.markdown("### 🔧 Data Filters")
    age_range = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))
    bmi_range = st.sidebar.slider("BMI Range", float(df["bmi"].min()), float(df["bmi"].max()),
                                   (float(df["bmi"].min()), float(df["bmi"].max())), 0.1)
    sex_filter = st.sidebar.multiselect("Gender", df["sex"].unique().tolist(),
                                         default=df["sex"].unique().tolist())
    smoker_filter = st.sidebar.multiselect("Tobacco Use", df["smoker"].unique().tolist(),
                                            default=df["smoker"].unique().tolist())
    region_filter = st.sidebar.multiselect("Region (India)", df["region_india"].unique().tolist(),
                                            default=df["region_india"].unique().tolist())

    filtered = df[
        (df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) &
        (df["bmi"] >= bmi_range[0]) & (df["bmi"] <= bmi_range[1]) &
        (df["sex"].isin(sex_filter)) &
        (df["smoker"].isin(smoker_filter)) &
        (df["region_india"].isin(region_filter))
    ]

    st.subheader(f"📋 Filtered Data — {len(filtered)} of {len(df)} records")
    display_df = filtered[["age", "sex", "bmi", "children", "smoker", "region_india", "charges_inr", "risk"]].copy()
    display_df.columns = ["Age", "Gender", "BMI", "Children", "Smoker", "Region", "Charges (₹)", "Risk"]
    display_df["Risk"] = display_df["Risk"].map({0: "Low", 1: "High"})
    st.dataframe(display_df, use_container_width=True)

    # Stats in INR
    st.subheader("📈 Descriptive Statistics")
    stats_df = filtered[["age", "bmi", "children", "charges_inr"]].describe().round(2)
    stats_df.columns = ["Age", "BMI", "Children", "Charges (₹)"]
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visualizations")

    viz_type = st.selectbox("Select Chart",
        ["BMI Distribution (Indian Standard)", "Charges Distribution (₹)", "BMI vs Charges (₹)",
         "Smoker Distribution", "Region-wise Charges (₹)", "Age vs Charges (₹)",
         "Correlation Heatmap", "Box Plot: Smoker vs Charges (₹)"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if viz_type == "BMI Distribution (Indian Standard)":
        ax.hist(filtered["bmi"], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(23.0, color="orange", linestyle="--", linewidth=2, label="Indian Overweight (23.0)")
        ax.axvline(25.0, color="red", linestyle="--", linewidth=2, label="Indian Obese (25.0)")
        ax.axvline(filtered["bmi"].mean(), color="green", linestyle="-.", label=f"Mean: {filtered['bmi'].mean():.1f}")
        ax.set_xlabel("BMI"); ax.set_ylabel("Frequency")
        ax.set_title("BMI Distribution (Indian/Asian Standard Thresholds)"); ax.legend()

    elif viz_type == "Charges Distribution (₹)":
        ax.hist(filtered["charges_inr"], bins=30, color="seagreen", edgecolor="black", alpha=0.7)
        ax.axvline(RISK_THRESHOLD_INR, color="red", linestyle="--", linewidth=2,
                   label=f"Risk Threshold (₹{RISK_THRESHOLD_INR:,.0f})")
        ax.set_xlabel("Charges (₹)"); ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Medical Charges (₹)"); ax.legend()

    elif viz_type == "BMI vs Charges (₹)":
        for status, color in [("yes", "red"), ("no", "green")]:
            subset = filtered[filtered["smoker"] == status]
            ax.scatter(subset["bmi"], subset["charges_inr"], c=color, alpha=0.5,
                       label=f"Smoker: {status}", edgecolors="black", linewidth=0.3)
        ax.set_xlabel("BMI"); ax.set_ylabel("Charges (₹)")
        ax.set_title("BMI vs Charges in ₹ (by Smoking Status)"); ax.legend()

    elif viz_type == "Smoker Distribution":
        counts = filtered["smoker"].value_counts()
        ax.pie(counts, labels=["Non-Smoker", "Smoker"], autopct="%1.1f%%",
               colors=["#2ecc71", "#e74c3c"], startangle=90, explode=(0, 0.05), shadow=True)
        ax.set_title("Tobacco User vs Non-User Distribution")

    elif viz_type == "Region-wise Charges (₹)":
        region_avg = filtered.groupby("region_india")["charges_inr"].mean().sort_values(ascending=False)
        colors_list = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"][:len(region_avg)]
        bars = ax.bar(region_avg.index, region_avg.values, color=colors_list, edgecolor="black")
        for bar, val in zip(bars, region_avg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                    f"₹{val:,.0f}", ha="center", fontweight="bold", fontsize=9)
        ax.set_xlabel("Region (India)"); ax.set_ylabel("Average Charges (₹)")
        ax.set_title("Region-wise Average Medical Cost (₹)")

    elif viz_type == "Age vs Charges (₹)":
        scatter = ax.scatter(filtered["age"], filtered["charges_inr"], c=filtered["risk"],
                             cmap="RdYlGn_r", alpha=0.6, edgecolors="black", linewidth=0.3)
        ax.axhline(y=RISK_THRESHOLD_INR, color="red", linestyle="--", linewidth=2,
                   label=f"Risk Threshold (₹{RISK_THRESHOLD_INR:,.0f})")
        ax.set_xlabel("Age"); ax.set_ylabel("Charges (₹)")
        ax.set_title("Age vs Charges in ₹ (by Risk Level)"); ax.legend()
        plt.colorbar(scatter, ax=ax, label="Risk (0=Low, 1=High)")

    elif viz_type == "Correlation Heatmap":
        plt.close(); fig, ax = plt.subplots(figsize=(8, 6))
        numeric = filtered[["age", "bmi", "children", "charges_inr", "risk"]]
        numeric.columns = ["Age", "BMI", "Children", "Charges (₹)", "Risk"]
        sns.heatmap(numeric.corr().round(2), annot=True, cmap="coolwarm", center=0,
                    fmt=".2f", ax=ax, square=True, linewidths=0.5)
        ax.set_title("Correlation Heatmap")

    elif viz_type == "Box Plot: Smoker vs Charges (₹)":
        plt.close(); fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="smoker", y="charges_inr", data=filtered, ax=ax,
                    palette={"yes": "#e74c3c", "no": "#2ecc71"})
        ax.set_xlabel("Tobacco Use"); ax.set_ylabel("Charges (₹)")
        ax.set_title("Tobacco Use vs Medical Charges (₹)")

    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Accuracy")
    st.markdown("Detailed evaluation of the Neural Network classifier.")
    st.markdown("---")

    if not models_loaded:
        st.error("⚠️ Models not found!"); st.stop()

    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, precision_score, recall_score, f1_score)

    # Prepare evaluation data
    df_eval = df.copy()
    df_eval["sex_encoded"] = df_eval["sex"].map({"male": 1, "female": 0})
    df_eval["smoker_encoded"] = df_eval["smoker"].map({"yes": 1, "no": 0})
    region_dummies = pd.get_dummies(df_eval["region"], prefix="region", dtype=int)
    df_eval = pd.concat([df_eval, region_dummies], axis=1)

    all_features = [
        "age", "bmi", "children", "sex_encoded", "smoker_encoded",
        "region_northeast", "region_northwest", "region_southeast", "region_southwest",
    ]
    X_eval = df_eval[all_features].values
    y_eval = df_eval["risk"].values
    X_eval_scaled = scaler.transform(X_eval)
    y_pred_eval = mlp_model.predict(X_eval_scaled)
    y_proba_eval = mlp_model.predict_proba(X_eval_scaled)

    acc = accuracy_score(y_eval, y_pred_eval)
    prec = precision_score(y_eval, y_pred_eval)
    rec = recall_score(y_eval, y_pred_eval)
    f1 = f1_score(y_eval, y_pred_eval)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc * 100:.2f}%")
    c2.metric("Precision", f"{prec * 100:.2f}%")
    c3.metric("Recall", f"{rec * 100:.2f}%")
    c4.metric("F1 Score", f"{f1 * 100:.2f}%")

    st.markdown("---")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred_eval)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix — MLP Neural Network")
    st.pyplot(fig_cm)
    plt.close()

    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | True Positives (High Risk correctly identified) | **{cm[1][1]}** |
    | True Negatives (Low Risk correctly identified) | **{cm[0][0]}** |
    | False Positives (wrongly flagged High Risk) | **{cm[0][1]}** |
    | False Negatives (missed High Risk) | **{cm[1][0]}** |
    """)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_eval, y_pred_eval,
                                    target_names=["Low Risk", "High Risk"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    # Architecture
    st.subheader("Neural Network Architecture")
    st.markdown("""
    | Layer | Neurons | Activation |
    |-------|---------|------------|
    | Input | 9 features | — |
    | Hidden 1 | 64 | ReLU |
    | Hidden 2 | 32 | ReLU |
    | Hidden 3 | 16 | ReLU |
    | Output | 1 | Sigmoid |

    **Optimizer:** Adam | **Learning Rate:** Adaptive (initial 0.001) | **Early Stopping:** Yes
    """)

    # Loss curve
    if hasattr(mlp_model, "loss_curve_"):
        st.subheader("Training Loss Curve")
        fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
        ax_loss.plot(mlp_model.loss_curve_, color="steelblue", linewidth=2)
        ax_loss.set_xlabel("Iterations"); ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Neural Network Training Loss Curve")
        ax_loss.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — REGIONAL ANALYSIS (India Map)
# ══════════════════════════════════════════════════════════════════════
elif page == "🗺️ Regional Analysis":
    st.title("🗺️ Regional Healthcare Analysis — India 🇮🇳")
    st.markdown("Healthcare metrics mapped to Indian regions.")
    st.markdown("---")

    region_stats = df.groupby("region_india").agg(
        avg_charges_inr=("charges_inr", "mean"),
        avg_bmi=("bmi", "mean"),
        avg_age=("age", "mean"),
        total_patients=("charges", "count"),
        high_risk_pct=("risk", "mean"),
        smoker_pct=("smoker", lambda x: (x == "yes").mean()),
    ).round(2)

    st.subheader("📊 Region-wise Statistics (₹)")
    display_stats = region_stats.copy()
    display_stats["avg_charges_inr"] = display_stats["avg_charges_inr"].apply(lambda x: f"₹{x:,.0f}")
    display_stats["high_risk_pct"] = (display_stats["high_risk_pct"] * 100).apply(lambda x: f"{x:.1f}%")
    display_stats["smoker_pct"] = (display_stats["smoker_pct"] * 100).apply(lambda x: f"{x:.1f}%")
    display_stats["avg_bmi"] = display_stats["avg_bmi"].apply(lambda x: f"{x:.1f}")
    display_stats["avg_age"] = display_stats["avg_age"].apply(lambda x: f"{x:.1f}")
    display_stats.columns = ["Avg Cost (₹)", "Avg BMI", "Avg Age", "Patients", "High Risk %", "Tobacco %"]
    st.dataframe(display_stats, use_container_width=True)

    # India map
    st.subheader("🗺️ Interactive Map — India")
    india_coords = {
        "North India": [28.6139, 77.2090],   # Delhi
        "South India": [13.0827, 80.2707],   # Chennai
        "East India": [22.5726, 88.3639],    # Kolkata
        "West India": [19.0760, 72.8777],    # Mumbai
    }

    map_data = []
    for region_name in region_stats.index:
        coords = india_coords[region_name]
        map_data.append({"lat": coords[0], "lon": coords[1]})
    st.map(pd.DataFrame(map_data), latitude="lat", longitude="lon", size=300)

    # Region details
    st.subheader("📍 Region Details")
    for region_name, row in region_stats.iterrows():
        city = {"North India": "Delhi", "South India": "Chennai",
                "East India": "Kolkata", "West India": "Mumbai"}[region_name]
        with st.expander(f"📍 {region_name} (Representative City: {city})"):
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Avg Cost", f"₹{row['avg_charges_inr']:,.0f}")
            rc2.metric("High Risk", f"{row['high_risk_pct']*100:.1f}%")
            rc3.metric("Patients", int(row["total_patients"]))

    # Comparison charts
    st.subheader("📊 Region Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    region_avg = df.groupby("region_india")["charges_inr"].mean().sort_values(ascending=False)
    bars = axes[0].bar(region_avg.index, region_avg.values,
                color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"], edgecolor="black")
    for bar, val in zip(bars, region_avg.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                     f"₹{val:,.0f}", ha="center", fontweight="bold", fontsize=8)
    axes[0].set_title("Average Charges by Region (₹)", fontweight="bold")
    axes[0].set_ylabel("Charges (₹)")

    region_bmi = df.groupby("region_india")["bmi"].mean().sort_values(ascending=False)
    bars2 = axes[1].bar(region_bmi.index, region_bmi.values,
                color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"], edgecolor="black")
    axes[1].axhline(y=23.0, color="orange", linestyle="--", label="Indian Overweight (23)")
    axes[1].axhline(y=25.0, color="red", linestyle="--", label="Indian Obese (25)")
    axes[1].set_title("Average BMI by Region (Indian Standard)", fontweight="bold")
    axes[1].set_ylabel("BMI"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── Footer ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("🇮🇳 **Smart Healthcare Risk Prediction System**")
st.sidebar.markdown("Indian Edition • Amounts in ₹ (INR)")
st.sidebar.markdown("Built with Streamlit, Scikit-learn & Neural Networks")
#stsrt forntend and the streamlit run app.py
#  cd healthcare-risk-project\frontend