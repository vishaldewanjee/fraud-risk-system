import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Risk Detection",
    page_icon="🔐",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

# Store transaction history during the current session
if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []


# ---------------- STYLING ----------------
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    background: #F0F4F8;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #334155 !important;
}

[data-testid="stMetricValue"] {
    color: #0F172A !important;
}

[data-testid="stMetricDelta"] {
    color: #334155 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔐 Fraud Risk System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Transaction Check",
        "📊 Analytics",
        "🧠 Model Insights",
        "📈 Model Performance"
    ]
)

# ---------------- TRANSACTION CHECK ----------------
if page == "🏠 Transaction Check":

    st.title("🔐 Fraud Risk Detection")
    st.write(
        "Analyze a transaction to see the fraud risk predicted "
        "by the machine-learning model."
    )

    st.info(
        "Demo mode: the model requires anonymized transaction "
        "features. If you do not have those values, the app uses "
        "zeros for them. This is for software testing only."
    )

    st.subheader("💳 Transaction Details")

    with st.form("transaction_form"):

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=10.0,
                help="Enter the transaction amount."
            )

        with col2:
            transaction_time = st.number_input(
                "Transaction Time",
                min_value=0.0,
                value=1000.0,
                step=1.0,
                help="Enter the Time value expected by your model."
            )


        feature_values = {}

        with st.expander("Advanced: Anonymized Model Features (V1–V28)"):
            st.caption(
                "Enter the V1–V28 values expected by your trained model."
            )

            for start in range(1, 29, 4):
                cols = st.columns(4)

                for offset, col in enumerate(cols):
                    number = start + offset

                    with col:
                        feature_values[f"V{number}"] = st.number_input(
                            f"V{number}",
                            value=0.0,
                            step=0.1,
                            format="%.4f",
                            key=f"feature_V{number}"
                        )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction",
            type="primary",
            use_container_width=True
        )

    if submitted:

        # Build the exact payload expected by Transaction
        payload = {
            "Time": float(transaction_time),
            **feature_values,
            "Amount": float(amount)
        }

        st.write("Number of fields:", len(payload))

        with st.spinner("Analyzing transaction..."):

            try:
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=15
                )

                if response.status_code == 200:

                    result = response.json()

                    probability = result["fraud_probability"] * 100
                    risk = result["risk_level"]
                    decision = result["decision"]

                    
                    # Save the prediction to session history
                    st.session_state.transaction_history.append({
                        "Amount": amount,
                        "Time": transaction_time,
                        "Fraud Probability (%)": round(probability, 2),
                        "Risk Level": risk,
                        "Decision": decision
                    })

                    st.divider()
                    st.subheader("📋 Analysis Result")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Predicted Fraud Probability",
                        f"{probability:.6f}%"
                    )

                    col2.metric(
                        "Risk Level",
                        risk
                    )

                    col3.metric(
                        "Model Decision",
                        decision
                    )

                    if risk == "Low":
                        st.success(
                            "The model assigned this transaction a low risk score."
                        )
                    elif risk == "Medium":
                        st.info(
                            "The model assigned this transaction a medium risk score."
                        )
                    elif risk == "High":
                        st.warning(
                            "The model flagged this transaction for manual review."
                        )
                    else:
                        st.error(
                            "The model assigned this transaction a critical risk score."
                        )

                    st.caption(
                        "This is a model prediction, not a guarantee "
                        "that a transaction is legitimate or fraudulent."
                    )

                else:
                    st.error(
                        f"API returned status code {response.status_code}."
                    )
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. Make sure your FastAPI "
                    "server is running at http://127.0.0.1:8000."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The API took too long to respond. Please try again."
                )

            except (KeyError, ValueError):
                st.error(
                    "The API response was not in the expected format."
                )

            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")


# ---------------- ANALYTICS ----------------
    
elif page == "📊 Analytics":
    st.title("📊 Transaction Analytics")

    history = st.session_state.transaction_history

    if history:
        import pandas as pd

        df = pd.DataFrame(history)

        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions Analyzed", len(df))
        col2.metric(
            "High/Critical Risk",
            int(df["Risk Level"].isin(["High", "Critical"]).sum())
        )
        col3.metric(
            "Average Fraud Probability",
            f'{df["Fraud Probability (%)"].mean():.2f}%'
        )

        st.subheader("Transaction History")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv,
            file_name="fraud_transaction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.transaction_history = []
            st.rerun()

    else:
        st.info(
            "No transactions analyzed yet. Go to Transaction Check "
            "and analyze a transaction to build your history."
        )




# ---------------- MODEL INSIGHTS ----------------
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")
    st.info("We'll add model explanations in a later step.")


# ---------------- MODEL PERFORMANCE ----------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.info("We'll add evaluation metrics and charts in a later step.")

