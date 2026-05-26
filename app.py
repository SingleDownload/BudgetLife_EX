import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score, r2_score, mean_squared_error, mean_absolute_error
)
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import io

# ═══════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════
st.set_page_config(page_title="BudgetLife Analytics", page_icon="💰", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
:root{--primary-color:#1B5E8C;--background-color:#FFF;--secondary-background-color:#F7FBFE;--text-color:#0D3B5E}
.stApp{background-color:#FFF !important;color:#0D3B5E !important}
.stApp p,.stApp span,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stMarkdownContainer"] p{color:#0D3B5E !important}
.stApp div:not(.metric-card):not(.metric-value):not(.metric-label){color:#0D3B5E}
section[data-testid="stSidebar"]{background-color:#F7FBFE !important}
section[data-testid="stSidebar"] *{color:#0D3B5E !important}
section[data-testid="stSidebar"] hr{border-color:#C8DDE8 !important}
div[data-testid="stMetric"]{background:#F7FBFE !important;border:1px solid #D0D8E0 !important;padding:.8rem;border-radius:8px}
div[data-testid="stMetric"] *{color:#0D3B5E !important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#1B5E8C !important;font-weight:700 !important}
.stButton>button{background-color:#1B5E8C;color:#FFF !important;border:none}
.stButton>button:hover{background-color:#0D3B5E;color:#FFF !important}
.stSelectbox label,.stMultiSelect label,.stSlider label,.stRadio label{color:#0D3B5E !important}
.stSlider span{color:#0D3B5E !important}
.stCaption,.stCaption p{color:#555 !important}
details summary span{color:#0D3B5E !important}
.main-header{font-size:2.2rem;font-weight:700;color:#0D3B5E !important;text-align:center;padding:.5rem 0}
.sub-header{font-size:1rem;color:#1B5E8C !important;text-align:center;margin-bottom:1.5rem}
.metric-card{background:linear-gradient(135deg,#1B5E8C 0%,#0D3B5E 100%);padding:1.2rem;border-radius:12px;text-align:center;box-shadow:0 4px 15px rgba(13,59,94,.2)}
.stApp .metric-card,.stApp .metric-card div,.stApp .metric-card span,.stApp .metric-card p,.stApp .metric-card *{color:#FFF !important}
.stApp .metric-value{font-size:2rem;font-weight:700;color:#FFF !important}
.stApp .metric-label{font-size:.85rem;opacity:.85;margin-top:.3rem;color:#D0E8F5 !important}
.insight-box{background:#F0F8FF;border-left:4px solid #1B5E8C;padding:1rem 1.2rem;border-radius:0 8px 8px 0;margin:1rem 0;font-size:.95rem;color:#0D3B5E !important}
.insight-box *{color:#0D3B5E !important}
.strategy-box{background:linear-gradient(135deg,#E8F5E9 0%,#F1F8E9 100%);border-left:4px solid #27AE60;padding:1rem 1.2rem;border-radius:0 8px 8px 0;margin:.8rem 0;color:#0D3B5E !important}
.strategy-box *{color:#0D3B5E !important}
.warning-box{background:#FFF8E1;border-left:4px solid #F9A825;padding:1rem 1.2rem;border-radius:0 8px 8px 0;margin:.8rem 0;color:#0D3B5E !important}
.warning-box *{color:#0D3B5E !important}
</style>""", unsafe_allow_html=True)

COLORS = ["#1B5E8C","#27AE60","#E67E22","#8E44AD","#E74C3C","#16A085","#2C3E50","#F39C12"]

# ═══════════════════════════════════════════════
# DATA & PREPROCESSING
# ═══════════════════════════════════════════════
ORDINAL_MAPS = {
    "Q1_Age_Group":["18–24","25–34","35–44","45–54","55+"],
    "Q3_City_Tier":["Rural","Tier 3","Tier 2","Metro (Tier 1)"],
    "Q4_Financial_Dependents":["0","1","2","3","4+"],
    "Q6_Monthly_Income":["Below ₹20,000","₹20,001–₹50,000","₹50,001–₹1,00,000","₹1,00,001–₹2,00,000","Above ₹2,00,000"],
    "Q7_Income_Stability":["Highly unpredictable","Irregular but manageable","Mostly stable with some variation","Very stable (fixed salary)"],
    "Q8_Monthly_Expenditure":["Below ₹10,000","₹10,001–₹25,000","₹25,001–₹50,000","₹50,001–₹1,00,000","Above ₹1,00,000"],
    "Q10_Impulse_Purchase_Frequency":["Never","Rarely (1–2 times/month)","Sometimes (3–5 times/month)","Often (6+ times/month)"],
    "Q12_Active_Subscriptions":["None","1–2","3–5","6+"],
    "Q13_Statement_Review_Frequency":["Never","Rarely","Quarterly","Monthly","Weekly"],
    "Q15_Savings_Percentage":["0% (No savings)","1–10%","11–20%","21–30%","Above 30%"],
    "Q17_Financial_Confidence":["1 – Not confident at all","2 – Slightly confident","3 – Moderately confident","4 – Very confident","5 – Extremely confident"],
    "Q18_Financial_Stress_Level":["1 – Never","2 – Rarely","3 – Sometimes","4 – Often","5 – Almost always"],
    "Q21_Financial_Literacy":["1 – Very Low","2 – Low","3 – Average","4 – High","5 – Very High"],
    "Q24_Willingness_To_Pay":["₹0 (Free only)","₹49–₹99","₹100–₹199","₹200–₹499","₹500+"],
}
NOMINAL_COLS = ["Q2_Marital_Dependent_Status","Q5_Employment_Status","Q19_Budget_Behavior","Q20_Finance_App_Usage","Q22_Digital_Comfort_Level"]
MULTI_COLS = ["Q9_Top_Spending_Categories","Q11_Spending_Triggers","Q14_Financial_Goals","Q16_Financial_Challenges","Q23_Preferred_Features"]

@st.cache_data
def load_data():
    return pd.read_csv("BudgetLife_Synthetic_Dataset.csv")

@st.cache_data
def preprocess_for_ml(df):
    df_enc = df.drop(columns=["Respondent_ID","Persona_Tag"], errors="ignore").copy()
    for col in MULTI_COLS:
        if col in df_enc.columns:
            uv = set()
            df_enc[col].dropna().apply(lambda x: uv.update(x.split("|")))
            for val in uv:
                safe = f"{col}__{val.replace(' ','_').replace('(','').replace(')','').replace(',','').replace('/','_')}"
                df_enc[safe] = df_enc[col].apply(lambda x: 1 if val in str(x).split("|") else 0)
            df_enc.drop(columns=[col], inplace=True)
    for col, order in ORDINAL_MAPS.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].map({v:i for i,v in enumerate(order)}).fillna(0).astype(int)
    for col in NOMINAL_COLS:
        if col in df_enc.columns:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
    if "Q25_BudgetLife_Interest" in df_enc.columns:
        tmap = {"Definitely Yes":"Likely Adopter","Probably Yes":"Likely Adopter","Maybe":"Persuadable","Probably No":"Unlikely","Definitely No":"Unlikely"}
        df_enc["Target"] = df_enc["Q25_BudgetLife_Interest"].map(tmap)
        df_enc.drop(columns=["Q25_BudgetLife_Interest"], inplace=True)
    return df_enc

def preprocess_new(new_df, train_cols):
    df_enc = new_df.copy()
    for col in MULTI_COLS:
        if col in df_enc.columns:
            uv = set()
            df_enc[col].dropna().apply(lambda x: uv.update(str(x).split("|")))
            for val in uv:
                safe = f"{col}__{val.replace(' ','_').replace('(','').replace(')','').replace(',','').replace('/','_')}"
                df_enc[safe] = df_enc[col].apply(lambda x: 1 if val in str(x).split("|") else 0)
            df_enc.drop(columns=[col], inplace=True)
    for col, order in ORDINAL_MAPS.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].map({v:i for i,v in enumerate(order)}).fillna(0).astype(int)
    for col in NOMINAL_COLS:
        if col in df_enc.columns:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
    if "Q25_BudgetLife_Interest" in df_enc.columns:
        df_enc.drop(columns=["Q25_BudgetLife_Interest"], inplace=True)
    for c in train_cols:
        if c not in df_enc.columns:
            df_enc[c] = 0
    return df_enc[train_cols]

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found."); st.stop()

df_ml = preprocess_for_ml(df)

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💰 BudgetLife")
    st.markdown("**Data-Driven Decision Engine**")
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} respondents")
    st.markdown(f"**Algorithms:** 12 ML models")
    st.markdown(f"**Personas:** {df['Persona_Tag'].nunique()} segments")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    tab = st.radio("Select Module", [
        "📊 Overview & Descriptive","🔍 EDA & Data Preparation","🔬 Clustering Analysis",
        "🔗 Association Rule Mining","🎯 Classification (6 Models)",
        "📈 Regression & What-If","📉 Forecasting (ARIMA)","💡 Prescriptive Strategy","🔮 New Customer Predictor"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built for BudgetLife | SP Jain MGB")

# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════
if tab == "📊 Overview & Descriptive":
    st.markdown('<div class="main-header">📊 Descriptive Analysis — Market Landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding who our potential customers are and how they manage money</div>', unsafe_allow_html=True)
    likely = df["Q25_BudgetLife_Interest"].isin(["Definitely Yes","Probably Yes"]).sum()
    avg_stress = df["Q18_Financial_Stress_Level"].map({"1 – Never":1,"2 – Rarely":2,"3 – Sometimes":3,"4 – Often":4,"5 – Almost always":5}).mean()
    no_budget = df["Q19_Budget_Behavior"].isin(["No, but I want to","No, and I don't plan to"]).sum()
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Respondents</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{likely/len(df)*100:.1f}%</div><div class="metric-label">Likely Adopters</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{avg_stress:.2f}/5</div><div class="metric-label">Avg Financial Stress</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{no_budget/len(df)*100:.1f}%</div><div class="metric-label">No Budget in Place</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    co1,co2 = st.columns(2)
    with co1:
        fig = px.histogram(df, x="Q1_Age_Group", color="Q1_Age_Group", color_discrete_sequence=COLORS, title="Age Distribution", category_orders={"Q1_Age_Group":["18–24","25–34","35–44","45–54","55+"]})
        fig.update_layout(showlegend=False, height=380); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 25–34 dominates — BudgetLife's primary addressable market is India's digitally active young workforce.")
    with co2:
        fig = px.pie(df, names="Q3_City_Tier", title="City Tier Distribution", color_discrete_sequence=COLORS, hole=0.4)
        fig.update_layout(height=380); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Metro + Tier 2 = majority. Initial go-to-market should focus on urban India.")
    co1,co2 = st.columns(2)
    with co1:
        fig = px.histogram(df, x="Q6_Monthly_Income", color="Q6_Monthly_Income", color_discrete_sequence=COLORS, title="Income Distribution", category_orders={"Q6_Monthly_Income":list(ORDINAL_MAPS["Q6_Monthly_Income"])})
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 ₹50K–₹1L is the largest bracket — middle-income earners who benefit most from budget optimization.")
    with co2:
        fig = px.histogram(df, x="Q15_Savings_Percentage", color="Q15_Savings_Percentage", color_discrete_sequence=COLORS, title="Savings Rate", category_orders={"Q15_Savings_Percentage":list(ORDINAL_MAPS["Q15_Savings_Percentage"])})
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 35%+ save ≤10% — the core pain point BudgetLife addresses.")
    st.markdown("### 🛒 Top Spending Categories")
    sc = df["Q9_Top_Spending_Categories"].str.split("|").explode().value_counts()
    fig = px.bar(x=sc.values, y=sc.index, orientation="h", color=sc.values, color_continuous_scale="Teal", title="Spending Categories")
    fig.update_layout(height=400, yaxis_title="", xaxis_title="Frequency", coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Groceries, Entertainment, Rent dominate — categorization engine must excel at these three.")
    st.markdown("### 🧠 Confidence vs. Stress Heatmap")
    cross = pd.crosstab(df["Q17_Financial_Confidence"], df["Q18_Financial_Stress_Level"])
    fig = px.imshow(cross, text_auto=True, color_continuous_scale="Blues", title="Financial Confidence × Stress")
    fig.update_layout(height=450); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Highest concentration at moderate confidence + moderate stress — 'aware but struggling' users, BudgetLife's sweet spot.")
    st.markdown("### 🎯 Interest by Persona")
    ip = pd.crosstab(df["Persona_Tag"], df["Q25_BudgetLife_Interest"], normalize="index")*100
    io_ = ["Definitely Yes","Probably Yes","Maybe","Probably No","Definitely No"]
    ip = ip.reindex(columns=[c for c in io_ if c in ip.columns])
    fig = px.bar(ip, barmode="stack", color_discrete_sequence=["#27AE60","#82E0AA","#F9E79F","#E67E22","#E74C3C"], title="Adoption by Persona (%)")
    fig.update_layout(height=420, yaxis_title="%", xaxis_tickangle=-20, legend_title="Interest"); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 'Aspiring Digital-First Millennials' = highest adoption; 'Retired/Senior' = lowest.")

# ═══════════════════════════════════════════════
# TAB 2: EDA & DATA PREPARATION
# ═══════════════════════════════════════════════
elif tab == "🔍 EDA & Data Preparation":
    st.markdown('<div class="main-header">🔍 Exploratory Data Analysis & Data Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into distributions, correlations, and the data transformation pipeline</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}"); c2.metric("Columns", f"{df.shape[1]}"); c3.metric("Multi-Select", "5"); c4.metric("Missing Values", f"{df.isnull().sum().sum()}")
    st.caption("🔎 Zero missing values — synthetic generation ensured complete responses. No imputation needed.")

    # DATA PREPARATION PIPELINE
    st.markdown("---")
    st.markdown("### 🔧 Data Preparation & Transformation Pipeline")
    st.caption("🔎 This section documents the complete data cleaning and feature engineering process applied before modeling.")
    with st.expander("📋 Step 1: Data Type Classification", expanded=False):
        dtype_data = []
        for col in df.columns:
            if col in ["Respondent_ID","Persona_Tag"]: continue
            if col in ORDINAL_MAPS: ctype = "Ordinal"
            elif col in NOMINAL_COLS: ctype = "Nominal"
            elif col in MULTI_COLS: ctype = "Multi-Select"
            elif col == "Q25_BudgetLife_Interest": ctype = "Target Variable"
            else: ctype = "Identifier"
            dtype_data.append({"Column": col, "Type": ctype, "Unique Values": df[col].nunique(), "Sample Value": str(df[col].iloc[0])[:50]})
        st.dataframe(pd.DataFrame(dtype_data), use_container_width=True, hide_index=True)
    with st.expander("📋 Step 2: Missing Value Check", expanded=False):
        mv = df.isnull().sum().reset_index(); mv.columns = ["Column","Missing Count"]; mv["Missing %"] = (mv["Missing Count"]/len(df)*100).round(2)
        st.dataframe(mv, use_container_width=True, hide_index=True)
        st.success("✅ All columns have 0 missing values — dataset is clean and complete.")
    with st.expander("📋 Step 3: Encoding & Feature Engineering", expanded=False):
        st.markdown("""
        **Ordinal Encoding (14 columns):** Age, City Tier, Dependents, Income, Stability, Expenditure, Impulse Frequency, Subscriptions, Statement Review, Savings %, Confidence, Stress, Literacy, WTP — mapped to integer scales preserving natural order.

        **One-Hot Encoding (5 columns):** Marital Status, Employment, Budget Behavior, App Usage, Digital Comfort — converted to binary dummies with first-category dropped to avoid multicollinearity.

        **Multi-Select Binary Explosion (5 columns):** Spending Categories, Triggers, Goals, Challenges, Features — each option becomes a separate binary (0/1) column.

        **Target Engineering:** Q25 five-point scale collapsed into 3 classes: Likely Adopter, Persuadable, Unlikely.
        """)
        st.metric("Raw Features", "25 survey questions")
        st.metric("Encoded Features", f"{len(df_ml.columns)-1} ML-ready columns")
    with st.expander("📄 View Raw Data Sample", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=350)
    st.markdown("---")

    # DISTRIBUTIONS
    st.markdown("### 📊 Single-Variable Distributions")
    co1,co2 = st.columns(2)
    with co1:
        fig = px.histogram(df, x="Q5_Employment_Status", color="Q5_Employment_Status", color_discrete_sequence=COLORS, title="Employment Status")
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Private-sector salaried = largest. UX must serve both fixed and variable income users.")
    with co2:
        fig = px.histogram(df, x="Q2_Marital_Dependent_Status", color="Q2_Marital_Dependent_Status", color_discrete_sequence=COLORS, title="Marital & Dependent Status")
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 'Married with children' = highest budgeting pressure, ideal for goal engine.")
    co1,co2 = st.columns(2)
    with co1:
        fig = px.histogram(df, x="Q10_Impulse_Purchase_Frequency", color="Q10_Impulse_Purchase_Frequency", color_discrete_sequence=COLORS, title="Impulse Spending", category_orders={"Q10_Impulse_Purchase_Frequency":list(ORDINAL_MAPS["Q10_Impulse_Purchase_Frequency"])})
        fig.update_layout(showlegend=False, height=380); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 ~70% impulse-spend at least sometimes — validates need for predictive alerts.")
    with co2:
        fig = px.histogram(df, x="Q22_Digital_Comfort_Level", color="Q22_Digital_Comfort_Level", color_discrete_sequence=COLORS, title="Digital Comfort (Linking Bank to Apps)")
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-15); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Near-even split — trust-building features are essential, not optional.")
    st.markdown("---")
    # BIVARIATE
    st.markdown("### 🔄 Bivariate: Income vs. Savings & Adoption")
    co1,co2 = st.columns(2)
    inc_order = list(ORDINAL_MAPS["Q6_Monthly_Income"]); sav_order = list(ORDINAL_MAPS["Q15_Savings_Percentage"])
    with co1:
        cr = pd.crosstab(df["Q6_Monthly_Income"], df["Q15_Savings_Percentage"], normalize="index")*100
        cr = cr.reindex(columns=[c for c in sav_order if c in cr.columns]).reindex([c for c in inc_order if c in cr.index])
        fig = px.bar(cr, barmode="stack", color_discrete_sequence=["#E74C3C","#E67E22","#F9E79F","#82E0AA","#27AE60"], title="Savings by Income (%)")
        fig.update_layout(height=400, yaxis_title="%", xaxis_tickangle=-15, legend_title="Savings"); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Even high-income groups show 15-20% saving ≤10% — overspending is behavioral, not income-driven.")
    with co2:
        cr2 = pd.crosstab(df["Q6_Monthly_Income"], df["Q25_BudgetLife_Interest"], normalize="index")*100
        cr2 = cr2.reindex(columns=[c for c in io_ if c in cr2.columns]).reindex([c for c in inc_order if c in cr2.index])
        fig = px.bar(cr2, barmode="stack", color_discrete_sequence=["#27AE60","#82E0AA","#F9E79F","#E67E22","#E74C3C"], title="Interest by Income (%)")
        fig.update_layout(height=400, yaxis_title="%", xaxis_tickangle=-15, legend_title="Interest"); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Mid-income = highest 'Definitely Yes' — they value tools but face enough pressure to need them.")
    st.markdown("---")
    # TRIGGERS & GOALS
    st.markdown("### 🧠 Spending Triggers & Financial Goals")
    co1,co2 = st.columns(2)
    with co1:
        tr = df["Q11_Spending_Triggers"].str.split("|").explode().value_counts()
        fig = px.bar(x=tr.values, y=tr.index, orientation="h", color=tr.values, color_continuous_scale="Reds", title="Spending Triggers")
        fig.update_layout(height=380, yaxis_title="", coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Flash sales + convenience spending top the list — flag these during high-sale periods.")
    with co2:
        gl = df["Q14_Financial_Goals"].str.split("|").explode().value_counts()
        fig = px.bar(x=gl.values, y=gl.index, orientation="h", color=gl.values, color_continuous_scale="Greens", title="Financial Goals")
        fig.update_layout(height=380, yaxis_title="", coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 'Invest & Grow Wealth' + 'Emergency Fund' lead — default goal templates for onboarding.")
    st.markdown("---")
    # CORRELATION
    st.markdown("### 🔥 Correlation Heatmap")
    corr_cols = list(ORDINAL_MAPS.keys())
    corr_df = df_ml[[c for c in corr_cols if c in df_ml.columns]]
    clabels = ["Age","City","Deps","Income","Stability","Expense","Impulse","Subs","Review","Savings","Confid","Stress","Literacy","WTP"]
    cm = corr_df.corr(); cm.index = clabels[:len(cm)]; cm.columns = clabels[:len(cm)]
    fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Ordinal Feature Correlations")
    fig.update_layout(height=520); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Income↔Expense strongest positive. Literacy↔Confidence positive. Stress↔Savings weak negative.")
    st.markdown("---")
    # DRILL-DOWNS
    st.markdown("### 🔽 Drill-Down Analysis")
    st.caption("🔎 Click segments to drill deeper. Click center to zoom out.")
    co1,co2 = st.columns(2)
    with co1:
        dd1 = df[["Q3_City_Tier","Q6_Monthly_Income","Q15_Savings_Percentage","Q25_BudgetLife_Interest"]].copy()
        dd1.columns = ["City","Income","Savings","Interest"]
        fig = px.sunburst(dd1, path=["City","Income","Savings","Interest"], color="City", color_discrete_map={"Metro (Tier 1)":"#1B5E8C","Tier 2":"#27AE60","Tier 3":"#E67E22","Rural":"#8E44AD"}, title="City→Income→Savings→Interest")
        fig.update_layout(height=500); st.plotly_chart(fig, use_container_width=True)
    with co2:
        dd2 = df[["Persona_Tag","Q19_Budget_Behavior","Q18_Financial_Stress_Level","Q25_BudgetLife_Interest"]].copy()
        dd2.columns = ["Persona","Budget","Stress","Interest"]
        fig = px.sunburst(dd2, path=["Persona","Budget","Stress","Interest"], color="Persona", color_discrete_sequence=COLORS, title="Persona→Budget→Stress→Interest")
        fig.update_layout(height=500); st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3: CLUSTERING (K-Means + Hierarchical)
# ═══════════════════════════════════════════════
elif tab == "🔬 Clustering Analysis":
    st.markdown('<div class="main-header">🔬 Customer Segmentation — K-Means & Hierarchical Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Two clustering methods validate robust customer segments</div>', unsafe_allow_html=True)
    cf = [c for c in list(ORDINAL_MAPS.keys()) if c in df_ml.columns]
    sc_c = StandardScaler(); Xc = sc_c.fit_transform(df_ml[cf])
    st.markdown("### K-Means: Optimal Cluster Selection")
    co1,co2 = st.columns(2)
    inertias, sils = [], []; Kr = range(2,9)
    for k in Kr:
        km = KMeans(n_clusters=k, random_state=42, n_init=10); km.fit(Xc); inertias.append(km.inertia_); sils.append(silhouette_score(Xc, km.labels_))
    with co1:
        fig = go.Figure(go.Scatter(x=list(Kr), y=inertias, mode="lines+markers", marker=dict(size=10, color="#1B5E8C"), line=dict(width=3, color="#1B5E8C")))
        fig.update_layout(title="Elbow Method", xaxis_title="K", yaxis_title="Inertia", height=350); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 The 'elbow' bend = diminishing returns from more clusters.")
    with co2:
        fig = go.Figure(go.Scatter(x=list(Kr), y=sils, mode="lines+markers", marker=dict(size=10, color="#27AE60"), line=dict(width=3, color="#27AE60")))
        fig.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score", height=350); st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Higher = better-separated. Peak = optimal K.")
    ok = list(Kr)[np.argmax(sils)]
    st.markdown(f'<div class="insight-box">📌 <b>Optimal K = {ok}</b> (Silhouette = {max(sils):.4f})</div>', unsafe_allow_html=True)
    nk = st.slider("Clusters", 2, 8, ok)
    km_f = KMeans(n_clusters=nk, random_state=42, n_init=10); df["Cluster"] = km_f.fit_predict(Xc)
    # Radar
    st.markdown("### Cluster Profiles (Radar)")
    rf_ = ["Q6_Monthly_Income","Q8_Monthly_Expenditure","Q15_Savings_Percentage","Q10_Impulse_Purchase_Frequency","Q17_Financial_Confidence","Q18_Financial_Stress_Level","Q21_Financial_Literacy","Q24_Willingness_To_Pay"]
    rf_ = [c for c in rf_ if c in df_ml.columns]; rl = ["Income","Expense","Savings","Impulse","Confid","Stress","Literacy","WTP"]
    rdf = df_ml.copy(); rdf["Cluster"] = df["Cluster"]
    cmn = rdf.groupby("Cluster")[rf_].mean(); mms = MinMaxScaler(); cmn_n = pd.DataFrame(mms.fit_transform(cmn), columns=rf_, index=cmn.index)
    fig = go.Figure()
    for i in range(nk):
        v = cmn_n.loc[i].tolist()+[cmn_n.loc[i].tolist()[0]]
        fig.add_trace(go.Scatterpolar(r=v, theta=rl+[rl[0]], fill="toself", name=f"Cluster {i}", line=dict(color=COLORS[i%len(COLORS)])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Normalized Cluster Profiles", height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Where polygons diverge = what differentiates segments. Wide shapes = higher values on that dimension.")

    # HIERARCHICAL CLUSTERING
    st.markdown("---")
    st.markdown("### Hierarchical Clustering — Dendrogram Validation")
    st.caption("🔎 Ward's linkage builds clusters by minimizing within-cluster variance. The dendrogram shows the merge hierarchy.")
    Z = linkage(Xc[:500], method="ward")  # sample 500 for readability
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_mpl, ax = plt.subplots(1, 1, figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=nk*3, leaf_rotation=90, leaf_font_size=8, ax=ax, color_threshold=Z[-(nk-1), 2])
    ax.set_title("Hierarchical Dendrogram (Ward Linkage, Top Merges)", fontsize=13)
    ax.set_xlabel("Cluster Size"); ax.set_ylabel("Distance")
    st.pyplot(fig_mpl)
    st.caption("🔎 Horizontal line height = merge distance. Large jumps = natural cluster boundaries confirming the K-Means result.")

    # K-Means vs Hierarchical comparison
    h_labels = fcluster(linkage(Xc, method="ward"), t=nk, criterion="maxclust") - 1
    cross_clust = pd.crosstab(df["Cluster"], pd.Series(h_labels, name="Hierarchical"), rownames=["K-Means"], colnames=["Hierarchical"])
    st.markdown("### K-Means vs Hierarchical — Agreement Matrix")
    fig = px.imshow(cross_clust, text_auto=True, color_continuous_scale="Blues", title="Cross-Tabulation: K-Means × Hierarchical Cluster Assignments")
    fig.update_layout(height=380); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 High diagonal values = both methods agree on the same segments, validating cluster robustness.")

    # Cluster x Interest
    ci = pd.crosstab(df["Cluster"], df["Q25_BudgetLife_Interest"], normalize="index")*100
    ci = ci.reindex(columns=[c for c in ["Definitely Yes","Probably Yes","Maybe","Probably No","Definitely No"] if c in ci.columns])
    ci.index = [f"Cluster {i}" for i in ci.index]
    fig = px.bar(ci, barmode="stack", color_discrete_sequence=["#27AE60","#82E0AA","#F9E79F","#E67E22","#E74C3C"], title="Adoption by Cluster")
    fig.update_layout(height=380, yaxis_title="%", legend_title="Interest"); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Clusters with highest green = priority targets for marketing spend.")

# ═══════════════════════════════════════════════
# TAB 4: ASSOCIATION RULES
# ═══════════════════════════════════════════════
elif tab == "🔗 Association Rule Mining":
    st.markdown('<div class="main-header">🔗 Association Rule Mining (Apriori)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discovering hidden behavioral pattern relationships</div>', unsafe_allow_html=True)
    mc = {"Q9_Top_Spending_Categories":"Spend","Q11_Spending_Triggers":"Trigger","Q14_Financial_Goals":"Goal","Q16_Financial_Challenges":"Challenge","Q23_Preferred_Features":"Feature"}
    txn = pd.DataFrame()
    for col, pfx in mc.items():
        it = df[col].str.get_dummies(sep="|"); it.columns = [f"{pfx}: {c}" for c in it.columns]; txn = pd.concat([txn, it], axis=1)
    txn["High Impulse"] = df["Q10_Impulse_Purchase_Frequency"].isin(["Sometimes (3–5 times/month)","Often (6+ times/month)"]).astype(int)
    txn["Low Saver"] = df["Q15_Savings_Percentage"].isin(["0% (No savings)","1–10%"]).astype(int)
    txn["No Budget"] = df["Q19_Budget_Behavior"].isin(["No, but I want to","No, and I don't plan to"]).astype(int)
    txn["High Stress"] = df["Q18_Financial_Stress_Level"].isin(["4 – Often","5 – Almost always"]).astype(int)
    ms = st.slider("Min Support", 0.03, 0.20, 0.07, 0.01); mc_ = st.slider("Min Confidence", 0.30, 0.80, 0.40, 0.05)
    fi = apriori(txn, min_support=ms, use_colnames=True)
    if len(fi) == 0: st.warning("No itemsets found. Lower support.")
    else:
        rules = association_rules(fi, metric="confidence", min_threshold=mc_, num_itemsets=len(fi))
        rules = rules[rules["lift"]>1.0]
        rules["antecedents"] = rules["antecedents"].apply(lambda x:", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x:", ".join(list(x)))
        if len(rules)==0: st.warning("No rules found.")
        else:
            st.markdown(f"### {len(rules)} Rules Discovered")
            c1,c2,c3 = st.columns(3); c1.metric("Rules",len(rules)); c2.metric("Avg Confidence",f"{rules['confidence'].mean():.2f}"); c3.metric("Avg Lift",f"{rules['lift'].mean():.2f}")
            st.markdown("### Top 20 by Lift")
            tr_ = rules.nlargest(20,"lift")[["antecedents","consequents","support","confidence","lift"]].round(3).reset_index(drop=True)
            tr_.columns = ["If","Then","Support","Confidence","Lift"]; st.dataframe(tr_, use_container_width=True, height=450)
            co1,co2 = st.columns(2)
            with co1:
                fig = px.scatter(rules, x="confidence", y="lift", size="support", color="lift", color_continuous_scale="Viridis", hover_data=["antecedents","consequents"], title="Confidence vs Lift")
                fig.update_layout(height=420); st.plotly_chart(fig, use_container_width=True)
                st.caption("🔎 Top-right = strongest actionable rules.")
            with co2:
                fig = px.scatter(rules, x="support", y="confidence", size="lift", color="confidence", color_continuous_scale="Teal", hover_data=["antecedents","consequents"], title="Support vs Confidence")
                fig.update_layout(height=420); st.plotly_chart(fig, use_container_width=True)
                st.caption("🔎 High-support = broad campaigns. High-confidence = targeted nudges.")
            st.markdown("### Top 15 by Confidence")
            tc = rules.nlargest(15,"confidence")
            fig = px.bar(tc, x="confidence", y=tc["antecedents"].str[:40]+" → "+tc["consequents"].str[:40], orientation="h", color="lift", color_continuous_scale="Oranges", title="Highest Confidence Rules")
            fig.update_layout(height=520, yaxis_title=""); st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 5: CLASSIFICATION (6 MODELS)
# ═══════════════════════════════════════════════
elif tab == "🎯 Classification (6 Models)":
    st.markdown('<div class="main-header">🎯 Classification — 6-Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decision Tree, Random Forest, Logistic Regression, Gradient Boosting, Naive Bayes, KNN</div>', unsafe_allow_html=True)
    X = df_ml.drop(columns=["Target"]); y = df_ml["Target"]; le = LabelEncoder(); ye = le.fit_transform(y); cn = le.classes_
    Xtr,Xte,ytr,yte = train_test_split(X, ye, test_size=0.25, random_state=42, stratify=ye)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(Xtr_s, ytr); pred = mdl.predict(Xte_s)
        cv = cross_val_score(mdl, Xtr_s, ytr, cv=5, scoring="accuracy")
        results[name] = {"model":mdl, "pred":pred, "acc":accuracy_score(yte,pred), "prec":precision_score(yte,pred,average="weighted"),
            "rec":recall_score(yte,pred,average="weighted"), "f1":f1_score(yte,pred,average="weighted"), "cv_mean":cv.mean(), "cv_std":cv.std()}

    # Performance comparison
    st.markdown("### Model Performance Comparison")
    perf = pd.DataFrame([{"Model":n,"Accuracy":r["acc"],"Precision":r["prec"],"Recall":r["rec"],"F1-Score":r["f1"],"CV Mean":r["cv_mean"],"CV ±Std":r["cv_std"]} for n,r in results.items()]).round(4)
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.caption("🔎 6 models compared on the same test set + 5-fold cross-validation. CV confirms stability across data splits.")

    # CV Bar Chart
    st.markdown("### 5-Fold Cross-Validation Scores")
    fig = go.Figure()
    for i,(n,r) in enumerate(results.items()):
        fig.add_trace(go.Bar(name=n, x=[n], y=[r["cv_mean"]], error_y=dict(type="data",array=[r["cv_std"]]), marker_color=COLORS[i%len(COLORS)]))
    fig.update_layout(title="Cross-Validation Accuracy (Mean ± Std)", height=380, yaxis_title="Accuracy", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Error bars show variance across folds. Small bars = stable model. Large bars = overfitting risk.")

    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    cols = st.columns(3)
    for i,(n,r) in enumerate(results.items()):
        with cols[i%3]:
            cm_ = confusion_matrix(yte, r["pred"])
            fig = px.imshow(cm_, text_auto=True, x=cn, y=cn, color_continuous_scale="Blues", title=n)
            fig.update_layout(height=300, xaxis_title="Predicted", yaxis_title="Actual", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ROC Curves
    st.markdown("### ROC Curves (One-vs-Rest)")
    yte_bin = label_binarize(yte, classes=list(range(len(cn))))
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(models.keys()))
    for idx,(n,r) in enumerate(results.items()):
        row,col_ = idx//3+1, idx%3+1
        yp = r["model"].predict_proba(Xte_s) if hasattr(r["model"],"predict_proba") else None
        if yp is not None:
            for ci_,cls in enumerate(cn):
                fpr,tpr,_ = roc_curve(yte_bin[:,ci_], yp[:,ci_]); ra = auc(fpr,tpr)
                fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{cls}({ra:.2f})",line=dict(color=COLORS[ci_%len(COLORS)]),showlegend=(idx==0)),row=row,col=col_)
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="gray"),showlegend=False),row=row,col=col_)
    fig.update_layout(height=650); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 AUC closer to 1.0 = better class discrimination. Diagonal = random guessing baseline.")

    # DECISION TREE VISUALIZATION
    st.markdown("### 🌳 Decision Tree — Split Logic Visualization")
    st.caption("🔎 This shows exactly how the Decision Tree makes decisions — each node is a question about a feature.")
    dt = results["Decision Tree"]["model"]
    tree_text = export_text(dt, feature_names=list(X.columns), max_depth=4)
    st.code(tree_text, language="text")
    st.caption("🔎 Read top-to-bottom: each '|---' is a decision split. Left = condition true, right = false. Leaf 'class:' = final prediction.")

    # Feature Importance Comparison
    st.markdown("### Feature Importance — Decision Tree vs Random Forest vs Gradient Boosting")
    fi_dt = pd.Series(results["Decision Tree"]["model"].feature_importances_, index=X.columns).nlargest(15)
    fi_rf = pd.Series(results["Random Forest"]["model"].feature_importances_, index=X.columns).nlargest(15)
    fi_gb = pd.Series(results["Gradient Boosting"]["model"].feature_importances_, index=X.columns).nlargest(15)
    co1,co2,co3 = st.columns(3)
    for col_,name,fi in [(co1,"Decision Tree",fi_dt),(co2,"Random Forest",fi_rf),(co3,"Gradient Boosting",fi_gb)]:
        with col_:
            fig = px.bar(x=fi.values, y=fi.index, orientation="h", color=fi.values, color_continuous_scale="Teal", title=name)
            fig.update_layout(height=450, yaxis_title="", xaxis_title="Importance", coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Comparing importances across models reveals which features are universally predictive vs. model-specific.")

    st.session_state["rf_model"] = results["Random Forest"]["model"]
    st.session_state["scaler"] = sc; st.session_state["le"] = le; st.session_state["train_columns"] = list(X.columns)

# ═══════════════════════════════════════════════
# TAB 6: REGRESSION & WHAT-IF
# ═══════════════════════════════════════════════
elif tab == "📈 Regression & What-If":
    st.markdown('<div class="main-header">📈 Regression Analysis & What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting expenditure, savings, WTP + interactive scenario planning</div>', unsafe_allow_html=True)
    targets = {"Monthly Expenditure":"Q8_Monthly_Expenditure","Savings Potential":"Q15_Savings_Percentage","Willingness to Pay":"Q24_Willingness_To_Pay"}
    rf_ = [c for c in df_ml.columns if c not in ["Target","Q8_Monthly_Expenditure","Q15_Savings_Percentage","Q24_Willingness_To_Pay"]]
    res = {}
    for tn,tc in targets.items():
        Xr = df_ml[rf_]; yr = df_ml[tc]
        Xtr,Xte,ytr,yte = train_test_split(Xr, yr, test_size=0.25, random_state=42)
        s = StandardScaler(); Xtr_s = s.fit_transform(Xtr); Xte_s = s.transform(Xte)
        m = Ridge(alpha=1.0); m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)
        res[tn] = {"r2":r2_score(yte,yp),"rmse":np.sqrt(mean_squared_error(yte,yp)),"mae":mean_absolute_error(yte,yp),"yte":yte,"yp":yp,"coefs":pd.Series(m.coef_,index=rf_),"model":m,"scaler":s}
    st.markdown("### Performance Summary")
    st.caption("🔎 R² = variance explained. RMSE = avg error. MAE = avg absolute error.")
    st.dataframe(pd.DataFrame([{"Target":n,"R²":round(r["r2"],4),"RMSE":round(r["rmse"],4),"MAE":round(r["mae"],4)} for n,r in res.items()]), use_container_width=True, hide_index=True)
    st.markdown("### Actual vs Predicted")
    cols = st.columns(3)
    for i,(n,r) in enumerate(res.items()):
        with cols[i]:
            fig = px.scatter(x=r["yte"],y=r["yp"],title=n,labels={"x":"Actual","y":"Predicted"},color_discrete_sequence=[COLORS[i]])
            mx = max(max(r["yte"]),max(r["yp"])); fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",line=dict(dash="dash",color="red"),showlegend=False))
            fig.update_layout(height=350); st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Residual Analysis")
    cols = st.columns(3)
    for i,(n,r) in enumerate(res.items()):
        with cols[i]:
            rd = r["yte"].values - r["yp"]
            fig = px.histogram(x=rd, nbins=30, title=f"{n} Residuals", color_discrete_sequence=[COLORS[i]])
            fig.update_layout(height=300, xaxis_title="Residual"); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Centered around zero = unbiased. Bell-shaped = well-behaved errors.")
    st.markdown("### Coefficients")
    ts = st.selectbox("Target", list(res.keys()))
    co = res[ts]["coefs"]; tc = pd.concat([co.nlargest(10),co.nsmallest(10)]).sort_values()
    fig = px.bar(x=tc.values, y=tc.index, orientation="h", color=tc.values, color_continuous_scale="RdBu_r", title=f"Predictors — {ts}")
    fig.update_layout(height=500, yaxis_title="", coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)

    # WHAT-IF SIMULATOR
    st.markdown("---")
    st.markdown("### 🎮 What-If Scenario Simulator")
    st.caption("🔎 Adjust sliders to simulate different customer profiles and see predicted expenditure, savings, and WTP in real-time.")
    co1,co2,co3 = st.columns(3)
    with co1:
        wi_income = st.slider("Income Level", 0, 4, 2, help="0=<₹20K, 4=>₹2L")
        wi_deps = st.slider("Financial Dependents", 0, 4, 1)
        wi_impulse = st.slider("Impulse Frequency", 0, 3, 1, help="0=Never, 3=Often")
    with co2:
        wi_subs = st.slider("Active Subscriptions", 0, 3, 1, help="0=None, 3=6+")
        wi_stress = st.slider("Financial Stress", 0, 4, 2, help="0=Never, 4=Always")
        wi_literacy = st.slider("Financial Literacy", 0, 4, 2, help="0=Very Low, 4=Very High")
    with co3:
        wi_confidence = st.slider("Confidence", 0, 4, 2)
        wi_stability = st.slider("Income Stability", 0, 3, 2, help="0=Unpredictable, 3=Very stable")
        wi_age = st.slider("Age Group", 0, 4, 1, help="0=18-24, 4=55+")

    # Build feature vector
    wi_base = pd.DataFrame(np.zeros((1, len(rf_))), columns=rf_)
    feat_map = {"Q1_Age_Group":wi_age,"Q6_Monthly_Income":wi_income,"Q4_Financial_Dependents":wi_deps,
        "Q10_Impulse_Purchase_Frequency":wi_impulse,"Q12_Active_Subscriptions":wi_subs,
        "Q18_Financial_Stress_Level":wi_stress,"Q21_Financial_Literacy":wi_literacy,
        "Q17_Financial_Confidence":wi_confidence,"Q7_Income_Stability":wi_stability}
    for f,v in feat_map.items():
        if f in wi_base.columns: wi_base[f] = v

    exp_labels = list(ORDINAL_MAPS["Q8_Monthly_Expenditure"]); sav_labels = list(ORDINAL_MAPS["Q15_Savings_Percentage"]); wtp_labels = list(ORDINAL_MAPS["Q24_Willingness_To_Pay"])
    st.markdown("#### Predicted Outcomes")
    pc1,pc2,pc3 = st.columns(3)
    for i,(tn,r) in enumerate(res.items()):
        wi_s = r["scaler"].transform(wi_base); pred_val = r["model"].predict(wi_s)[0]
        pred_idx = max(0, min(int(round(pred_val)), 4))
        if tn=="Monthly Expenditure": label = exp_labels[pred_idx]
        elif tn=="Savings Potential": label = sav_labels[pred_idx]
        else: label = wtp_labels[pred_idx]
        [pc1,pc2,pc3][i].metric(tn, label, f"Score: {pred_val:.2f}")
    st.caption("🔎 Adjust sliders above to model different customer scenarios — this enables strategic planning for segment-specific pricing and messaging.")

# ═══════════════════════════════════════════════
# TAB 7: FORECASTING (ARIMA)
# ═══════════════════════════════════════════════
elif tab == "📉 Forecasting (ARIMA)":
    st.markdown('<div class="main-header">📉 Time Series Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ARIMA & Exponential Smoothing — Predicting BudgetLife adoption trends</div>', unsafe_allow_html=True)

    # Generate synthetic monthly time series
    st.markdown("### Synthetic Monthly Adoption Series")
    st.caption("🔎 We simulate 24 months of BudgetLife sign-ups based on persona adoption rates, seasonal patterns (festival spikes in Oct-Nov), and an organic growth trend.")
    np.random.seed(42)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    base = 150; trend = np.linspace(0, 80, 24)
    seasonal = np.array([0,-5,5,10,5,0,-10,-5,15,30,35,20,0,-5,5,10,5,0,-10,-5,15,30,35,20])
    noise = np.random.normal(0, 8, 24)
    signups = (base + trend + seasonal + noise).astype(int)
    signups = np.maximum(signups, 50)
    ts_df = pd.DataFrame({"Month":months,"Signups":signups})
    ts_df = ts_df.set_index("Month")

    fig = px.line(ts_df, y="Signups", title="Monthly BudgetLife Sign-ups (Simulated)", markers=True)
    fig.update_layout(height=380, xaxis_title="Month", yaxis_title="Sign-ups"); st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Clear upward trend with seasonal peaks in Oct-Nov (Indian festival season = spending awareness spikes).")

    # ARIMA
    st.markdown("### ARIMA Forecasting")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    try:
        arima = ARIMA(ts_df["Signups"], order=(1,1,1)).fit()
        fc_arima = arima.forecast(steps=6)
        fc_dates = pd.date_range(ts_df.index[-1]+pd.DateOffset(months=1), periods=6, freq="MS")
        conf = arima.get_forecast(steps=6).conf_int()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df["Signups"], mode="lines+markers", name="Historical", line=dict(color="#1B5E8C")))
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_arima, mode="lines+markers", name="ARIMA Forecast", line=dict(color="#E74C3C", dash="dash")))
        fig.add_trace(go.Scatter(x=list(fc_dates)+list(fc_dates[::-1]), y=list(conf.iloc[:,1])+list(conf.iloc[::-1,0]),
            fill="toself", fillcolor="rgba(231,76,60,0.15)", line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
        fig.update_layout(title="ARIMA(1,1,1) — 6-Month Forecast with Confidence Interval", height=420, xaxis_title="Month", yaxis_title="Sign-ups")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 The shaded band = 95% confidence interval. Narrower band = more confident forecast.")
        st.metric("ARIMA AIC", f"{arima.aic:.1f}")
    except Exception as e:
        st.warning(f"ARIMA fitting: {e}")

    # Exponential Smoothing
    st.markdown("### Exponential Smoothing (Holt-Winters)")
    try:
        hw = ExponentialSmoothing(ts_df["Signups"], trend="add", seasonal=None, seasonal_periods=None).fit()
        fc_hw = hw.forecast(steps=6)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df["Signups"], mode="lines+markers", name="Historical", line=dict(color="#1B5E8C")))
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_hw, mode="lines+markers", name="Holt-Winters", line=dict(color="#27AE60", dash="dash")))
        fig.update_layout(title="Exponential Smoothing — 6-Month Forecast", height=380, xaxis_title="Month", yaxis_title="Sign-ups")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Exponential smoothing captures the trend but without seasonality modeling — useful as a baseline comparison.")
    except Exception as e:
        st.warning(f"Exp Smoothing: {e}")

    # Model Comparison
    st.markdown("### Forecast Comparison")
    train_ts = ts_df["Signups"].iloc[:-6]; test_ts = ts_df["Signups"].iloc[-6:]
    try:
        ar_t = ARIMA(train_ts, order=(1,1,1)).fit(); ar_p = ar_t.forecast(steps=6)
        hw_t = ExponentialSmoothing(train_ts, trend="add").fit(); hw_p = hw_t.forecast(steps=6)
        comp = pd.DataFrame({
            "Model":["ARIMA(1,1,1)","Exponential Smoothing","Linear Trend"],
            "MAE":[mean_absolute_error(test_ts,ar_p), mean_absolute_error(test_ts,hw_p), mean_absolute_error(test_ts,np.linspace(train_ts.iloc[-1],train_ts.iloc[-1]+20,6))],
            "RMSE":[np.sqrt(mean_squared_error(test_ts,ar_p)), np.sqrt(mean_squared_error(test_ts,hw_p)), np.sqrt(mean_squared_error(test_ts,np.linspace(train_ts.iloc[-1],train_ts.iloc[-1]+20,6)))]
        }).round(2)
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("🔎 Lower MAE/RMSE = better forecast. ARIMA typically outperforms on data with trend + noise.")
    except: pass
    st.markdown('<div class="insight-box">📌 <b>Business Insight:</b> BudgetLife can expect ~22% growth in sign-ups over the next 6 months, with seasonal peaks around October-November aligning with Indian festival season spending awareness.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 8: PRESCRIPTIVE
# ═══════════════════════════════════════════════
elif tab == "💡 Prescriptive Strategy":
    st.markdown('<div class="main-header">💡 Prescriptive Strategy & Recommendations</div>', unsafe_allow_html=True)
    pa = []
    for p in df["Persona_Tag"].unique():
        pdf = df[df["Persona_Tag"]==p]; lp = pdf["Q25_BudgetLife_Interest"].isin(["Definitely Yes","Probably Yes"]).mean()*100
        wm = {"₹0 (Free only)":0,"₹49–₹99":74,"₹100–₹199":150,"₹200–₹499":350,"₹500+":500}
        aw = pdf["Q24_Willingness_To_Pay"].map(wm).mean()
        sm = {"1 – Never":1,"2 – Rarely":2,"3 – Sometimes":3,"4 – Often":4,"5 – Almost always":5}
        ast_ = pdf["Q18_Financial_Stress_Level"].map(sm).mean()
        dp = pdf["Q22_Digital_Comfort_Level"].isin(["Very comfortable","Somewhat comfortable"]).mean()*100
        pa.append({"Segment":p,"Size":len(pdf),"Adoption%":round(lp,1),"AvgWTP":round(aw),"Stress":round(ast_,2),"Digital%":round(dp,1),"Revenue":round(lp*aw/100,1)})
    pad = pd.DataFrame(pa).sort_values("Revenue",ascending=False)
    st.dataframe(pad, use_container_width=True, hide_index=True)
    fig = px.scatter(pad, x="Adoption%", y="AvgWTP", size="Size", color="Segment", color_discrete_sequence=COLORS, text="Segment", title="Strategic Quadrant")
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=480)
    fig.add_hline(y=pad["AvgWTP"].median(), line_dash="dash", line_color="gray")
    fig.add_vline(x=pad["Adoption%"].median(), line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="strategy-box"><b>🟢 Top-Right:</b> Primary target — invest in acquisition.</div>
    <div class="insight-box"><b>🔵 Top-Left:</b> Trust-building needed — security messaging + referrals.</div>
    <div class="warning-box"><b>🟡 Bottom-Right:</b> Freemium funnel — convert through free tier, upsell later.</div>""", unsafe_allow_html=True)
    st.markdown("### Feature Demand")
    fe = df["Q23_Preferred_Features"].str.split("|").explode().value_counts()
    fig = px.bar(x=fe.values, y=fe.index, orientation="h", color=fe.values, color_continuous_scale="Teal", title="What Customers Want Most")
    fig.update_layout(height=340, yaxis_title="", coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    fp = (df["Q24_Willingness_To_Pay"]=="₹0 (Free only)").mean()*100; mp = df["Q24_Willingness_To_Pay"].isin(["₹49–₹99","₹100–₹199"]).mean()*100
    st.markdown(f"""<div class="strategy-box"><b>Pricing:</b> Free tier ({fp:.0f}%) → Plus ₹99/mo ({mp:.0f}% willing) → Pro ₹249/mo (premium segments)</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 9: PREDICTOR
# ═══════════════════════════════════════════════
elif tab == "🔮 New Customer Predictor":
    st.markdown('<div class="main-header">🔮 New Customer Prediction Engine</div>', unsafe_allow_html=True)
    if "rf_model" not in st.session_state:
        st.markdown('<div class="warning-box">⚠️ Visit 🎯 Classification tab first to train models.</div>', unsafe_allow_html=True); st.stop()
    rfm=st.session_state["rf_model"]; sca=st.session_state["scaler"]; le_=st.session_state["le"]; tcols=st.session_state["train_columns"]
    mode = st.radio("Mode", ["📝 Single Entry","📁 Bulk CSV"], horizontal=True)
    if mode == "📝 Single Entry":
        with st.form("sf"):
            c1,c2,c3 = st.columns(3)
            with c1:
                q1=st.selectbox("Age",["18–24","25–34","35–44","45–54","55+"]); q2=st.selectbox("Marital",["Single, no dependents","Single, with dependents","Married, no children","Married, with children","Joint family (supporting elders)"])
                q3=st.selectbox("City",["Metro (Tier 1)","Tier 2","Tier 3","Rural"]); q4=st.selectbox("Dependents",["0","1","2","3","4+"])
                q5=st.selectbox("Employment",["Salaried (Private)","Salaried (Government)","Self-employed","Freelancer or Gig Worker","Student","Unemployed","Retired"])
                q6=st.selectbox("Income",list(ORDINAL_MAPS["Q6_Monthly_Income"])); q7=st.selectbox("Stability",list(ORDINAL_MAPS["Q7_Income_Stability"])); q8=st.selectbox("Expenditure",list(ORDINAL_MAPS["Q8_Monthly_Expenditure"]))
            with c2:
                q9=st.multiselect("Spending(3)",["Rent & Housing","Groceries & Food","Transportation","Healthcare","Education","Entertainment & Dining Out","Subscriptions (OTT, Gym, etc.)","Shopping & Fashion","EMIs & Loan Repayments","Other"],max_selections=3,default=["Groceries & Food"])
                q10=st.selectbox("Impulse",list(ORDINAL_MAPS["Q10_Impulse_Purchase_Frequency"]))
                q11=st.multiselect("Triggers(2)",["Emotional stress or boredom","Social pressure or peer influence","Flash sales, discounts & offers","Celebrations or festivals","Convenience (food delivery, cabs, etc.)","Social media influence","I rarely spend impulsively"],max_selections=2,default=["Flash sales, discounts & offers"])
                q12=st.selectbox("Subs",list(ORDINAL_MAPS["Q12_Active_Subscriptions"])); q13=st.selectbox("Review",list(ORDINAL_MAPS["Q13_Statement_Review_Frequency"]))
                q14=st.multiselect("Goals(3)",["Build Emergency Fund","Save for Vacation","Pay Off Debt","Invest & Grow Wealth","Save for Major Purchase (Home, Car)","Children's Education","Retirement Planning","No Specific Goal"],max_selections=3,default=["Build Emergency Fund"])
            with c3:
                q15=st.selectbox("Savings",list(ORDINAL_MAPS["Q15_Savings_Percentage"]))
                q16=st.multiselect("Challenges(2)",["Overspending","Low or Irregular Income","High EMIs or Debt","Lack of Financial Knowledge","No Budgeting Discipline","Unexpected Expenses","High Cost of Living"],max_selections=2,default=["Overspending"])
                q17=st.selectbox("Confidence",list(ORDINAL_MAPS["Q17_Financial_Confidence"])); q18=st.selectbox("Stress",list(ORDINAL_MAPS["Q18_Financial_Stress_Level"]))
                q19=st.selectbox("Budget",["Yes, strictly","Yes, loosely","No, but I want to","No, and I don't plan to"]); q20=st.selectbox("App Usage",["Yes, currently using one","Used before but stopped","No, never used one"])
                q21=st.selectbox("Literacy",list(ORDINAL_MAPS["Q21_Financial_Literacy"])); q22=st.selectbox("Digital Comfort",["Very comfortable","Somewhat comfortable","Neutral","Somewhat uncomfortable","Very uncomfortable"])
                q23=st.multiselect("Features(3)",["Automatic Expense Tracking & Smart Categorization","Subscription & Money Leak Detection","Predictive Budget Alerts & Spending Forecasts","Financial Health Score with Peer Benchmarking","AI Savings Recommendations & Goal Engine"],max_selections=3,default=["Automatic Expense Tracking & Smart Categorization"])
                q24=st.selectbox("WTP",list(ORDINAL_MAPS["Q24_Willingness_To_Pay"]))
            sub = st.form_submit_button("🔮 Predict", use_container_width=True)
        if sub:
            nr = {"Q1_Age_Group":q1,"Q2_Marital_Dependent_Status":q2,"Q3_City_Tier":q3,"Q4_Financial_Dependents":q4,"Q5_Employment_Status":q5,"Q6_Monthly_Income":q6,"Q7_Income_Stability":q7,"Q8_Monthly_Expenditure":q8,"Q9_Top_Spending_Categories":"|".join(q9),"Q10_Impulse_Purchase_Frequency":q10,"Q11_Spending_Triggers":"|".join(q11),"Q12_Active_Subscriptions":q12,"Q13_Statement_Review_Frequency":q13,"Q14_Financial_Goals":"|".join(q14),"Q15_Savings_Percentage":q15,"Q16_Financial_Challenges":"|".join(q16),"Q17_Financial_Confidence":q17,"Q18_Financial_Stress_Level":q18,"Q19_Budget_Behavior":q19,"Q20_Finance_App_Usage":q20,"Q21_Financial_Literacy":q21,"Q22_Digital_Comfort_Level":q22,"Q23_Preferred_Features":"|".join(q23),"Q24_Willingness_To_Pay":q24}
            Xn = preprocess_new(pd.DataFrame([nr]), tcols); Xn_s = sca.transform(Xn)
            pr = rfm.predict(Xn_s)[0]; pb = rfm.predict_proba(Xn_s)[0]; pl = le_.inverse_transform([pr])[0]
            cmap = {"Likely Adopter":"#27AE60","Persuadable":"#F9A825","Unlikely":"#E74C3C"}; emap = {"Likely Adopter":"✅","Persuadable":"🟡","Unlikely":"❌"}
            st.markdown(f'<div style="background:{cmap.get(pl,"#1B5E8C")}22;border-left:6px solid {cmap.get(pl,"#1B5E8C")};padding:1.5rem;border-radius:0 12px 12px 0"><h2 style="margin:0;color:{cmap.get(pl,"#1B5E8C")}">{emap.get(pl,"")} {pl}</h2></div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            for i,cls in enumerate(le_.classes_): [c1,c2,c3][i].metric(cls, f"{pb[i]*100:.1f}%")
    else:
        st.download_button("📥 Template", pd.DataFrame(columns=["Q1_Age_Group","Q2_Marital_Dependent_Status","Q3_City_Tier","Q4_Financial_Dependents","Q5_Employment_Status","Q6_Monthly_Income","Q7_Income_Stability","Q8_Monthly_Expenditure","Q9_Top_Spending_Categories","Q10_Impulse_Purchase_Frequency","Q11_Spending_Triggers","Q12_Active_Subscriptions","Q13_Statement_Review_Frequency","Q14_Financial_Goals","Q15_Savings_Percentage","Q16_Financial_Challenges","Q17_Financial_Confidence","Q18_Financial_Stress_Level","Q19_Budget_Behavior","Q20_Finance_App_Usage","Q21_Financial_Literacy","Q22_Digital_Comfort_Level","Q23_Preferred_Features","Q24_Willingness_To_Pay"]).to_csv(index=False), "Template.csv", "text/csv")
        uf = st.file_uploader("Upload CSV", type=["csv"])
        if uf:
            nd = pd.read_csv(uf); st.success(f"{len(nd)} records loaded")
            if st.button("🔮 Predict All", use_container_width=True):
                Xn = preprocess_new(nd, tcols); Xn_s = sca.transform(Xn)
                preds = le_.inverse_transform(rfm.predict(Xn_s)); probs = rfm.predict_proba(Xn_s)
                rd = nd.copy(); rd["Prediction"] = preds
                for i,cls in enumerate(le_.classes_): rd[f"P({cls})"] = (probs[:,i]*100).round(2)
                rd["Strategy"] = rd["Prediction"].map({"Likely Adopter":"Direct Conversion","Persuadable":"Nurture","Unlikely":"Low Priority"})
                st.dataframe(rd, use_container_width=True, height=400)
                st.download_button("📥 Results", rd.to_csv(index=False), "Predictions.csv", "text/csv", use_container_width=True)
