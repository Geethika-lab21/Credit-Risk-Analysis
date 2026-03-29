import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- STYLING (CSS) -----------------
st.markdown("""
<style>

/* Background */
.main, .stApp {
    background-color: #f5f0e6;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #e8dccb;
    color: #3e2f1c;
}

/* Fix ALL text visibility */
label, div, span, p {
    color: #3e2f1c !important;
    font-weight: 500;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #3e2f1c !important;
    font-weight: 700;
    text-align: center;
}

/* Cards */
.card {
    background-color: #e8dccb;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Primary Buttons */
.stButton>button {
    background-color: #8b6f47 !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    border: none !important;
}

/* Preserving Dashboard KPI Cards */
.kpi-card {
    background-color: #e8dccb;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    text-align: center;
    margin-bottom: 20px;
    color: #3e2f1c;
}
.kpi-title {
    font-size: 1.1em;
    font-weight: 600;
    margin-bottom: 5px;
    color: #6e5737;
}
.kpi-value {
    font-size: 2em;
    font-weight: bold;
    margin: 0;
}

</style>
""", unsafe_allow_html=True)

# Set base colors for matplotlib/seaborn
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#e8dccb", "figure.facecolor": "#f5f0e6"})
plt.rcParams['text.color'] = '#3e2f1c'
plt.rcParams['axes.labelcolor'] = '#3e2f1c'
plt.rcParams['xtick.color'] = '#3e2f1c'
plt.rcParams['ytick.color'] = '#3e2f1c'


# ----------------- DATA & MODEL LOADING -----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/loan_sample.csv")
        # Ensure loan_id exists (while preserving 'id' for exact lookup)
        if 'loan_id' not in df.columns:
            if 'id' in df.columns:
                df['loan_id'] = df['id']
            else:
                df['loan_id'] = range(len(df))
                
        # Handle string loan_status if needed for charts
        if 'loan_status' not in df.columns:
            # Fallback if loan_status doesn't exist, try to find something similar or mock
            # Often it's 'target', 'status', etc.
            if 'status' in df.columns:
                df.rename(columns={'status': 'loan_status'}, inplace=True)
            else:
                df['loan_status'] = np.random.choice(['Fully Paid', 'Charged Off'], size=len(df))

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

df = load_data()
model = load_model()

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    ["Dashboard", "Predict by Input", "Data Insights"]
)


# ----------------- PAGE 1: DASHBOARD -----------------
if page == "Dashboard":
    st.title("📊 Credit Risk Analytics Dashboard")
    st.markdown("---")

    if not df.empty:
        # Determine defaults
        # We assume 'Charged Off', 'Default' are defaulters. Others non-defaulters.
        def_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
        defaulters = df['loan_status'].isin(def_statuses).sum() if 'loan_status' in df.columns else 0
        total_loans = len(df)
        non_defaulters = total_loans - defaulters
        default_rate = (defaulters / total_loans * 100) if total_loans > 0 else 0

        # KPI CARDS
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Loans</div><div class="kpi-value">{total_loans:,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Defaulters</div><div class="kpi-value">{defaulters:,}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Non-Defaulters</div><div class="kpi-value">{non_defaulters:,}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Default Rate</div><div class="kpi-value">{default_rate:.2f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CHARTS
        col_chart1, col_chart2 = st.columns(2)

        # 1. Pie Chart: Loan Status
        with col_chart1:
            st.markdown("### Loan Status Distribution")
            if 'loan_status' in df.columns:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                status_counts = df['loan_status'].value_counts()
                # Colors: Beige/brown palette adapted for pie
                colors = ['#8b6f47', '#bda888', '#e8dccb', '#5c4629', '#d1bfae']
                ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)
            else:
                st.info("No loan_status column found.")

        # 2. Bar Chart: Avg Income by Loan Status
        with col_chart2:
            st.markdown("### Avg Income by Loan Status")
            if 'annual_inc' in df.columns and 'loan_status' in df.columns:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                avg_inc = df.groupby('loan_status')['annual_inc'].mean().sort_values(ascending=False).head(10).reset_index()
                sns.barplot(data=avg_inc, x='annual_inc', y='loan_status', ax=ax2, hue='loan_status', palette="dark:#8b6f47_r", legend=False)
                ax2.set_xlabel("Average Annual Income")
                ax2.set_ylabel("Loan Status")
                st.pyplot(fig2)
            else:
                st.info("Missing annual_inc or loan_status columns.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart3, col_chart4 = st.columns(2)

        # 3. Histogram: DTI Distribution
        with col_chart3:
            st.markdown("### Debt-to-Income (DTI) Distribution")
            if 'dti' in df.columns:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.histplot(df['dti'].dropna(), bins=30, kde=True, color="#8b6f47", ax=ax3)
                ax3.set_xlabel("DTI")
                st.pyplot(fig3)
            else:
                st.info("No DTI column found.")

        # 4. Boxplot: Interest Rate vs Loan Status
        with col_chart4:
            st.markdown("### Interest Rate vs Loan Status")
            if 'int_rate' in df.columns and 'loan_status' in df.columns:
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                # Only take top 5 statuses for clarity if many
                top_statuses = df['loan_status'].value_counts().head(5).index
                sns.boxplot(x='int_rate', y='loan_status', data=df[df['loan_status'].isin(top_statuses)], hue='loan_status', legend=False, palette="YlOrBr", ax=ax4)
                ax4.set_xlabel("Interest Rate (%)")
                ax4.set_ylabel("Loan Status")
                st.pyplot(fig4)
            else:
                st.info("Missing int_rate or loan_status columns.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 5. Correlation Heatmap
        st.markdown("### Correlation Heatmap (Selected Features)")
        important_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_util', 'revol_bal', 'open_acc', 'total_acc']
        available_cols = [c for c in important_cols if c in df.columns]
        
        if len(available_cols) > 1:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            # numeric_only to prevent errors if features contain string values
            corr = df[available_cols].corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="YlOrBr", fmt=".2f", ax=ax5, linewidths=0.5)
            st.pyplot(fig5)
        else:
            st.warning("Not enough important features available for correlation heatmap.")

    else:
        st.warning("Dataset is empty or could not be loaded. Please check data/loan_sample.csv")


# ----------------- PAGE 2: PREDICT BY INPUT -----------------
elif page == "Predict by Input":
    st.markdown("<h1 style='text-align:center;'>Credit Risk Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'> Loan Risk Prediction</h4>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
        annual_inc = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    
    with col2:
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.5, step=0.1)
        tenure = st.number_input("Loan Tenure (months)", min_value=1, value=36, step=1)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_btn = st.button("🔮 Predict Risk", use_container_width=True)

    if predict_btn:
        if model is None:
            st.error("Model file not found. Please ensure 'models/model.pkl' exists.")
        else:
            if annual_inc <= 0:
                st.error("Annual Income must be greater than 0")
                st.stop()
            
            if loan_amnt <= 0 or int_rate <= 0:
                st.error("Loan amount and interest rate must be positive")
                st.stop()

            if annual_inc < 1000:
                st.warning("Income seems very low, prediction may be inaccurate")

            r = int_rate / (12 * 100)
            n = tenure
            if r > 0:
                installment = loan_amnt * r * (1 + r)**n / ((1 + r)**n - 1)
            else:
                installment = loan_amnt / n

            st.write(f"Estimated Monthly Installment: ₹{installment:.2f}")

            monthly_income = annual_inc / 12
            dti = installment / monthly_income if monthly_income > 0 else 0

            st.write(f"Calculated DTI: {dti:.2f}")

            with st.spinner("Analyzing credit risk..."):
                revol_util = 40
                revol_bal = 10000
                open_acc = 5
                total_acc = 10

                features = [[
                    loan_amnt, int_rate, installment,
                    annual_inc, dti,
                    revol_util, revol_bal,
                    open_acc, total_acc
                ]]
                
                try:
                    # Provide columns to prevent sklearn input warnings while ensuring correct data type
                    features_df = pd.DataFrame(features, columns=['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_util', 'revol_bal', 'open_acc', 'total_acc'])
                    
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(features_df)[0][1]
                        
                        if prob < 0.30:
                            risk = "Low Risk"
                        elif prob <= 0.50:
                            risk = "Medium Risk"
                        else:
                            risk = "High Risk"

                        st.markdown("<br>", unsafe_allow_html=True)
                        if risk == "Low Risk":
                            st.success("✅ Low Risk")
                        elif risk == "Medium Risk":
                            st.warning("⚠️ Medium Risk")
                        else:
                            st.error("🚨 High Risk of Default")

                        st.info(f"Probability of Default: {prob:.2%}")
                    else:
                        prediction = model.predict(features_df)[0]
                        st.markdown("<br>", unsafe_allow_html=True)
                        if prediction == 1:
                            st.error("🚨 High Risk of Default")
                        else:
                            st.success("✅ Low Risk")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")





# ----------------- PAGE 4: DATA INSIGHTS -----------------
elif page == "Data Insights":
    st.title("💡 Deep Data Insights")
    st.markdown("Explore key metrics, risky profiles, and summary statistics.")
    st.markdown("---")

    if df.empty:
        st.error("Dataset not loaded.")
    else:
        # 1. Top Risky Loans (High DTI + High Interest Rate)
        st.markdown("### 🚨 Top Risky Loans")
        st.markdown("Borrowers with highest Debt-to-Income (DTI) and Interest Rates:")
        if 'dti' in df.columns and 'int_rate' in df.columns:
            risky = df.sort_values(by=['dti', 'int_rate'], ascending=[False, False]).head(10)
            cols_to_show = ['loan_id', 'loan_amnt', 'int_rate', 'dti', 'annual_inc', 'loan_status']
            cols_to_show = [c for c in cols_to_show if c in risky.columns]
            st.dataframe(risky[cols_to_show], use_container_width=True)
        else:
            st.info("Required columns (dti, int_rate) missing for this insight.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 2. High DTI Borrowers Summary
        st.markdown("### ⚠️ High DTI Borrowers (DTI > 30)")
        if 'dti' in df.columns:
            high_dti = df[df['dti'] > 30]
            st.write(f"Found **{len(high_dti):,}** accounts with DTI higher than 30.")
            if not high_dti.empty:
                cols2 = ['loan_id', 'dti', 'annual_inc', 'loan_amnt']
                cols2 = [c for c in cols2 if c in high_dti.columns]
                st.dataframe(high_dti[cols2].head(10), use_container_width=True)
        else:
            st.info("Required column (dti) missing.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Summary Statistics
        st.markdown("### 📈 Dataset Summary Statistics")
        with st.expander("View Full Summary Statistics (Numerical)", expanded=True):
            st.dataframe(df.describe().T, use_container_width=True)
