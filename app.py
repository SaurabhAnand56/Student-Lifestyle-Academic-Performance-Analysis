import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              accuracy_score, classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Digital Life Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES  (top navbar + cards + insight boxes)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* hide default sidebar toggle & hamburger */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── TOP NAVBAR ─────────────────────────────── */
.topnav {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 0 14px 0;
    flex-wrap: wrap;
}
.nav-btn {
    background: transparent;
    border: 1.5px solid #444;
    color: #ccc;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    cursor: pointer;
    text-decoration: none;
    transition: all .2s;
    white-space: nowrap;
}
.nav-btn:hover  { background:#7F77DD22; border-color:#7F77DD; color:#fff; }
.nav-btn.active { background:#7F77DD;   border-color:#7F77DD; color:#fff; font-weight:600; }

/* ── METRIC CARDS ───────────────────────────── */
.card-row { display:flex; gap:16px; margin:12px 0; flex-wrap:wrap; }
.card {
    flex:1; min-width:140px;
    background:#1e1e2e;
    border-radius:12px;
    padding:16px 20px;
    border-left:4px solid #7F77DD;
}
.card-val  { font-size:26px; font-weight:700; color:#fff; }
.card-lbl  { font-size:12px; color:#aaa; margin-top:2px; }

/* ── INSIGHT BOXES ──────────────────────────── */
.insight {
    background:#16213e;
    border-left:4px solid #2ECC71;
    padding:10px 16px;
    border-radius:0 8px 8px 0;
    margin:5px 0;
    font-size:14px;
    color:#ddd;
}

/* ── SECTION HEADING ────────────────────────── */
.sec { font-size:18px; font-weight:600; color:#7F77DD; margin:18px 0 6px 0; }

/* ── AUTHOR CARD ────────────────────────────── */
.author-card {
    display:flex; align-items:center; gap:20px;
    background:#1e1e2e; border-radius:14px;
    padding:20px 24px; margin:12px 0 20px 0;
    border:1px solid #333;
}
.author-card img { border-radius:50%; width:80px; height:80px; object-fit:cover; }
.author-name  { font-size:20px; font-weight:700; color:#fff; }
.author-title { font-size:13px; color:#aaa; margin:3px 0 8px 0; }
.badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:12px; font-weight:600; text-decoration:none; margin-right:6px;
}
.badge-gh { background:#333; color:#fff; }
.badge-li { background:#0A66C2; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — current page
# ─────────────────────────────────────────────────────────────────────────────
PAGES = ["Home", "EDA Overview", "EDA Relationships",
         "ML Regression", "ML Classification",
         "Feature Importance", "Clustering", "Predict Score"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVBAR  (rendered on every page)
# ─────────────────────────────────────────────────────────────────────────────
icons = ["🏠","📊","🔗","🤖","🏷️","🔍","🗂️","🔮"]
cols  = st.columns(len(PAGES))
for i, (col, pg, ic) in enumerate(zip(cols, PAGES, icons)):
    with col:
        active = "active" if st.session_state.page == pg else ""
        if st.button(f"{ic} {pg}", key=f"nav_{i}",
                     use_container_width=True,
                     type="primary" if st.session_state.page == pg else "secondary"):
            st.session_state.page = pg
            st.rerun()

st.markdown("<hr style='margin:6px 0 18px 0; border-color:#333'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA — load from GitHub raw URL
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_CSV = (
    "https://raw.githubusercontent.com/"
    "SaurabhAnand56/Student-Lifestyle-Academic-Performance-Analysis/"
    "main/dataset/student_digital_life.csv"
)

@st.cache_data(show_spinner="Loading dataset from GitHub...")
def load_data():
    return pd.read_csv(GITHUB_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# MODELS — train once and cache
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models...")
def train_all(_df):
    df_ml = _df.copy()
    le = LabelEncoder()
    for col in ['gender','mental_health_status','parent_education_level','internet_quality']:
        df_ml[col] = le.fit_transform(df_ml[col])

    df_ml['total_screen_time']    = (df_ml['smartphone_usage_hours'] + df_ml['social_media_hours']
                                     + df_ml['gaming_hours'] + df_ml['streaming_hours'])
    df_ml['study_efficiency']     = df_ml['study_hours_per_day'] / (df_ml['total_screen_time'] + 1)
    df_ml['academic_commitment']  = (df_ml['class_attendance_percent'] + df_ml['assignment_completion_percent']) / 2
    df_ml['wellbeing_score']      = df_ml['sleep_hours'] + df_ml['exercise_hours']
    df_ml['performance_category'] = pd.cut(df_ml['final_exam_score'],
        bins=[0,60,75,88,100], labels=['At Risk','Average','Good','Excellent'])

    features = [
        'age','study_hours_per_day','smartphone_usage_hours','social_media_hours',
        'gaming_hours','streaming_hours','sleep_hours','exercise_hours',
        'class_attendance_percent','assignment_completion_percent',
        'caffeine_intake_cups','motivation_level',
        'gender','mental_health_status','parent_education_level','internet_quality',
        'total_screen_time','study_efficiency','academic_commitment','wellbeing_score'
    ]

    X, y_reg, y_clf = df_ml[features], df_ml['final_exam_score'], df_ml['performance_category']

    Xtr,Xte,ytr,yte         = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    Xtr_c,Xte_c,ytr_c,yte_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    gb  = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
    rf  = RandomForestRegressor(n_estimators=100,     random_state=42).fit(Xtr, ytr)
    lr  = LinearRegression().fit(Xtr, ytr)
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                  class_weight='balanced').fit(Xtr_c, ytr_c)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    df_ml['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(Xs)

    return dict(df_ml=df_ml, features=features,
                Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                Xte_c=Xte_c, yte_c=yte_c,
                gb=gb, rf=rf, lr=lr, clf=clf, Xs=Xs)

df = load_data()
md = train_all(df)

# helper
def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

pg = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if pg == "Home":
    # Author card
    st.markdown("""
    <div class="author-card">
        <img src="https://github.com/SaurabhAnand56.png" alt="Saurabh Anand"/>
        <div>
            <div class="author-name">Saurabh Anand</div>
            <div class="author-title">Data Analyst &nbsp;|&nbsp; Python &bull; Power BI &bull; SQL &bull; Tableau</div>
            <a class="badge badge-gh" href="https://github.com/SaurabhAnand56" target="_blank">GitHub</a>
            <a class="badge badge-li" href="https://www.linkedin.com/in/saurabhanand56" target="_blank">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.title("🎓 Student Digital Life — ML Analysis")
    st.markdown("""
    This project explores how **digital habits** — screen time, gaming, sleep, study hours —
    affect students' **final exam scores**. Built on a dataset of **15,000 students, 18 features**
    using Python, Scikit-learn, and Streamlit.
    """)

    # Metrics
    screen_avg = (df['smartphone_usage_hours']+df['social_media_hours']
                  +df['gaming_hours']+df['streaming_hours']).mean()
    st.markdown(f"""
    <div class="card-row">
        <div class="card"><div class="card-val">{len(df):,}</div><div class="card-lbl">Total Students</div></div>
        <div class="card"><div class="card-val">{df['final_exam_score'].mean():.1f}</div><div class="card-lbl">Avg Exam Score</div></div>
        <div class="card"><div class="card-val">{df['study_hours_per_day'].mean():.1f} h</div><div class="card-lbl">Avg Study Hours/Day</div></div>
        <div class="card"><div class="card-val">{screen_avg:.1f} h</div><div class="card-lbl">Avg Screen Time/Day</div></div>
        <div class="card"><div class="card-val">18</div><div class="card-lbl">Features</div></div>
        <div class="card"><div class="card-val">R² 0.68</div><div class="card-lbl">Best Model Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Key findings
    st.markdown('<div class="sec">Key Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">📚 <b>Study hours</b> — #1 predictor of exam score (feature importance 49%, r = 0.64)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">📱 <b>Smartphone usage</b> — strongest negative predictor (r = −0.14)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">🤖 <b>Gradient Boosting</b> — best model: R² = 0.68, RMSE = 10.59</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">🧠 <b>Mental health</b> — Good health students score ~11 pts higher than Poor</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">👥 <b>K-Means clustering</b> — 4 distinct student behaviour profiles identified</div>', unsafe_allow_html=True)

    # Quick nav cards
    st.markdown('<div class="sec">Explore the Project</div>', unsafe_allow_html=True)
    r1 = st.columns(4)
    nav_cards = [
        ("📊 EDA Overview",       "Distributions of all variables",          "EDA Overview"),
        ("🔗 EDA Relationships",  "How habits relate to exam scores",        "EDA Relationships"),
        ("🤖 ML Regression",      "Predict exact exam score",                "ML Regression"),
        ("🏷️ ML Classification", "Classify performance category",           "ML Classification"),
    ]
    for col, (title, desc, target) in zip(r1, nav_cards):
        with col:
            if st.button(f"**{title}**\n\n{desc}", use_container_width=True, key=f"homenav_{target}"):
                st.session_state.page = target
                st.rerun()

    r2 = st.columns(4)
    nav_cards2 = [
        ("🔍 Feature Importance", "Which habits matter most?",               "Feature Importance"),
        ("🗂️ Clustering",        "Student behaviour segments",              "Clustering"),
        ("🔮 Predict My Score",  "Enter your habits → get your score",      "Predict Score"),
        ("📋 Sample Data",       "",                                         None),
    ]
    for col, (title, desc, target) in zip(r2, nav_cards2):
        with col:
            if target:
                if st.button(f"**{title}**\n\n{desc}", use_container_width=True, key=f"homenav2_{target}"):
                    st.session_state.page = target
                    st.rerun()

    st.markdown('<div class="sec">Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "EDA Overview":
    st.title("📊 EDA — Overview")
    st.markdown("Distribution of individual variables across 15,000 students.")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Age distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['age'], bins=9, color='#7F77DD', edgecolor='black', linewidth=0.4, ax=ax)
        ax.set_xlabel("Age"); clean(ax)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Gender distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        cnt = df['gender'].value_counts()
        ax.bar(cnt.index, cnt.values, color=['#E57373','#64B5F6'], edgecolor='black', linewidth=0.4)
        for i,v in enumerate(cnt.values): ax.text(i, v+50, f'{v:,}', ha='center', fontweight='bold')
        clean(ax); st.pyplot(fig); plt.close()

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("#### Mental health status")
        fig,ax = plt.subplots(figsize=(6,4))
        cnt = df['mental_health_status'].value_counts()
        ax.bar(cnt.index, cnt.values, color=['#2ECC71','#F39C12','#E74C3C'], edgecolor='black', linewidth=0.4)
        for i,v in enumerate(cnt.values): ax.text(i, v+50, f'{v:,}', ha='center', fontweight='bold')
        clean(ax); st.pyplot(fig); plt.close()
        st.info("Majority report Average or Good mental health.")

    with c4:
        st.markdown("#### Study hours per day")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['study_hours_per_day'], bins=25, color='#2ECC71', edgecolor='black', linewidth=0.3, ax=ax)
        ax.axvline(df['study_hours_per_day'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['study_hours_per_day'].mean():.1f} hrs")
        ax.legend(); ax.set_xlabel("Study Hours"); clean(ax)
        st.pyplot(fig); plt.close()

    c5,c6 = st.columns(2)
    with c5:
        st.markdown("#### Smartphone usage per day")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['smartphone_usage_hours'], bins=25, color='#E74C3C', edgecolor='black', linewidth=0.3, ax=ax)
        ax.axvline(df['smartphone_usage_hours'].mean(), color='blue', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['smartphone_usage_hours'].mean():.1f} hrs")
        ax.legend(); ax.set_xlabel("Phone Hours"); clean(ax)
        st.pyplot(fig); plt.close()

    with c6:
        st.markdown("#### Final exam score distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['final_exam_score'], bins=30, color='#F39C12', edgecolor='black', linewidth=0.3, ax=ax)
        ax.axvline(df['final_exam_score'].mean(),   color='red',  linestyle='--', linewidth=1.5, label=f"Mean: {df['final_exam_score'].mean():.1f}")
        ax.axvline(df['final_exam_score'].median(), color='blue', linestyle='--', linewidth=1.5, label=f"Median: {df['final_exam_score'].median():.1f}")
        ax.legend(); ax.set_xlabel("Exam Score"); clean(ax)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Full correlation heatmap")
    fig,ax = plt.subplots(figsize=(13,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.4, ax=ax, annot_kws={'size':8})
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# EDA RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "EDA Relationships":
    st.title("🔗 EDA — Relationships with Exam Score")

    def scatter_trend(col, color, ax, label):
        ax.scatter(df[col], df['final_exam_score'], alpha=0.15, s=6, color=color)
        m,b = np.polyfit(df[col], df['final_exam_score'], 1)
        xl  = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(xl, m*xl+b, 'r-', linewidth=2)
        ax.set_xlabel(label); ax.set_ylabel("Exam Score"); clean(ax)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Study hours vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_trend('study_hours_per_day','#7F77DD', ax, "Study Hours/Day")
        st.pyplot(fig); plt.close()
        st.success("Strong positive  r = 0.64")

    with c2:
        st.markdown("#### Smartphone usage vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_trend('smartphone_usage_hours','#E74C3C', ax, "Phone Hours/Day")
        st.pyplot(fig); plt.close()
        st.error("Negative  r = −0.14")

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("#### Class attendance vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_trend('class_attendance_percent','#2ECC71', ax, "Attendance %")
        st.pyplot(fig); plt.close()
        st.success("Positive  r = 0.14")

    with c4:
        st.markdown("#### Sleep hours vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_trend('sleep_hours','#3498DB', ax, "Sleep Hours/Day")
        st.pyplot(fig); plt.close()
        st.info("Modest positive  r = 0.07")

    c5,c6 = st.columns(2)
    with c5:
        st.markdown("#### Exam score by gender")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='gender', y='final_exam_score', data=df, palette=['#E57373','#64B5F6'], ax=ax)
        clean(ax); st.pyplot(fig); plt.close()

    with c6:
        st.markdown("#### Exam score by mental health")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='mental_health_status', y='final_exam_score', data=df,
                    order=['Good','Average','Poor'], palette=['#2ECC71','#F39C12','#E74C3C'], ax=ax)
        clean(ax); st.pyplot(fig); plt.close()
        st.info("Good mental health → higher scores on average.")

# ══════════════════════════════════════════════════════════════════════════════
# ML REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "ML Regression":
    st.title("🤖 ML — Exam Score Regression")
    st.markdown("Comparing 3 models to predict the continuous final exam score.")

    gb_p = md['gb'].predict(md['Xte'])
    rf_p = md['rf'].predict(md['Xte'])
    lr_p = md['lr'].predict(md['Xte'])

    results = pd.DataFrame({
        'Model':  ['Linear Regression','Random Forest','Gradient Boosting'],
        'R²':     [round(r2_score(md['yte'],p),4) for p in [lr_p,rf_p,gb_p]],
        'RMSE':   [round(np.sqrt(mean_squared_error(md['yte'],p)),2) for p in [lr_p,rf_p,gb_p]]
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Actual vs Predicted (Gradient Boosting)")
        fig,ax = plt.subplots(figsize=(6,5))
        ax.scatter(md['yte'], gb_p, alpha=0.25, color='steelblue', s=8)
        ax.plot([md['yte'].min(),md['yte'].max()],[md['yte'].min(),md['yte'].max()],'r--',linewidth=1.5,label='Perfect fit')
        ax.text(5,95,'R² = 0.68',fontsize=12,color='red',fontweight='bold')
        ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score"); ax.legend(); clean(ax)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### R² Score Comparison")
        fig,ax = plt.subplots(figsize=(6,5))
        bars = ax.bar(results['Model'], results['R²'], color=['#3498db','#2ecc71','#e74c3c'], edgecolor='black', linewidth=0.5)
        ax.set_ylim(0,1); ax.set_ylabel("R²")
        for bar,val in zip(bars,results['R²']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, str(val), ha='center', fontweight='bold')
        plt.xticks(rotation=10); clean(ax)
        st.pyplot(fig); plt.close()

    st.markdown("#### Residual Plot")
    residuals = md['yte'] - gb_p
    fig,ax = plt.subplots(figsize=(12,4))
    ax.scatter(gb_p, residuals, alpha=0.2, color='#7F77DD', s=8)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Predicted Score"); ax.set_ylabel("Residual"); clean(ax)
    st.pyplot(fig); plt.close()
    st.info("Residuals scattered randomly around 0 — no major systematic bias.")

# ══════════════════════════════════════════════════════════════════════════════
# ML CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "ML Classification":
    st.title("🏷️ ML — Performance Classifier")
    st.markdown("Random Forest classifying students: **At Risk / Average / Good / Excellent**")

    y_pred_c = md['clf'].predict(md['Xte_c'])
    acc = accuracy_score(md['yte_c'], y_pred_c)

    a,b,c = st.columns(3)
    a.metric("Accuracy", f"{acc:.2%}")
    b.metric("Classes", "4")
    c.metric("Test samples", f"{len(md['yte_c']):,}")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Class distribution")
        cnt = md['df_ml']['performance_category'].value_counts()
        fig,ax = plt.subplots(figsize=(6,4))
        ax.bar(cnt.index, cnt.values, color=['#e74c3c','#f39c12','#3498db','#2ecc71'], edgecolor='black', linewidth=0.5)
        for i,v in enumerate(cnt.values): ax.text(i, v+30, f'{v:,}', ha='center', fontweight='bold', fontsize=9)
        ax.set_ylabel("Students"); clean(ax)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Confusion matrix")
        fig,ax = plt.subplots(figsize=(6,5))
        cm = confusion_matrix(md['yte_c'], y_pred_c, labels=['At Risk','Average','Good','Excellent'])
        ConfusionMatrixDisplay(cm, display_labels=['At Risk','Average','Good','Excellent']).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(classification_report(md['yte_c'], y_pred_c, output_dict=True)).T.round(2), use_container_width=True)
    st.info("Best at identifying Excellent and At-Risk students. Average vs Good is hardest to separate — similar lifestyle habits.")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Which lifestyle habits drive exam performance the most?")

    imp_df = pd.DataFrame({'Feature': md['features'], 'Importance': md['rf'].feature_importances_})\
               .sort_values('Importance', ascending=True).tail(15)
    pos = ['study_hours_per_day','academic_commitment','study_efficiency','motivation_level','sleep_hours','wellbeing_score']
    neg = ['smartphone_usage_hours','total_screen_time','gaming_hours','social_media_hours','streaming_hours']
    bar_colors = ['#2ecc71' if f in pos else '#e74c3c' if f in neg else '#3498db' for f in imp_df['Feature']]

    fig,ax = plt.subplots(figsize=(10,8))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=bar_colors, edgecolor='black', linewidth=0.4)
    ax.set_xlabel("Feature Importance Score")
    ax.legend(handles=[Patch(facecolor='#2ecc71',label='Positive habits'),
                        Patch(facecolor='#e74c3c',label='Digital distractions'),
                        Patch(facecolor='#3498db',label='Other')], loc='lower right')
    clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Pearson correlation with exam score")
    num_cols = ['study_hours_per_day','smartphone_usage_hours','social_media_hours','gaming_hours',
                'sleep_hours','exercise_hours','motivation_level','class_attendance_percent',
                'assignment_completion_percent','final_exam_score']
    corr = df[num_cols].corr()['final_exam_score'].drop('final_exam_score').sort_values()
    fig,ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(corr.index, corr.values,
                   color=['#E74C3C' if x < 0 else '#2ECC71' for x in corr],
                   edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8); ax.set_xlabel("Pearson r")
    for bar,val in zip(bars,corr.values):
        ax.text(val+(0.005 if val>=0 else -0.005), bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', ha='left' if val>=0 else 'right', fontsize=9)
    clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close()
    st.success("study_hours_per_day leads in both correlation (r=0.64) and feature importance (49%).")

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Clustering":
    st.title("🗂️ Student Segmentation — K-Means")
    st.markdown("4 distinct student behaviour profiles identified from 15,000 records.")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Elbow method")
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(md['Xs']).inertia_ for k in range(2,11)]
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(range(2,11), inertias, 'o-', color='steelblue', linewidth=2, markersize=7)
        ax.axvline(4, color='red', linestyle='--', linewidth=1.5, label='K=4')
        ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.legend(); clean(ax)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Exam score by cluster")
        fig,ax = plt.subplots(figsize=(6,4))
        data = [md['df_ml'][md['df_ml']['cluster']==i]['final_exam_score'].values for i in range(4)]
        bp = ax.boxplot(data, patch_artist=True, labels=[f'Cluster {i}' for i in range(4)])
        for patch,col in zip(bp['boxes'],['#3498db','#2ecc71','#e74c3c','#f39c12']):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_ylabel("Exam Score"); clean(ax)
        st.pyplot(fig); plt.close()

    st.markdown("#### Cluster profiles")
    cp = md['df_ml'].groupby('cluster').agg(
        Students=('final_exam_score','count'),
        Avg_Score=('final_exam_score','mean'),
        Study_Hrs=('study_hours_per_day','mean'),
        Sleep_Hrs=('sleep_hours','mean'),
        Phone_Hrs=('smartphone_usage_hours','mean'),
        Motivation=('motivation_level','mean')
    ).round(2).reset_index()
    st.dataframe(cp, use_container_width=True, hide_index=True)

    st.markdown("#### PCA 2D visualisation")
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(md['Xs'])
    fig,ax = plt.subplots(figsize=(10,5))
    sc = ax.scatter(coords[:,0], coords[:,1], c=md['df_ml']['cluster'], cmap='tab10', alpha=0.35, s=6)
    plt.colorbar(sc, ax=ax, label='Cluster')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    clean(ax); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT MY SCORE
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Predict Score":
    st.title("🔮 Predict My Exam Score")
    st.markdown("Live prediction using **Gradient Boosting** (R²=0.68, RMSE=10.59). Move the sliders to match your habits.")

    c1,c2 = st.columns(2)
    with c1:
        age          = st.slider("Age", 17, 25, 21)
        study_hours  = st.slider("Study hours/day", 0.0, 12.0, 4.5, step=0.1)
        sleep_hours  = st.slider("Sleep hours/day", 3.5, 10.0, 7.0, step=0.1)
        exercise     = st.slider("Exercise hours/day", 0.0, 5.0, 1.2, step=0.1)
        motivation   = st.slider("Motivation (1–10)", 1.0, 10.0, 6.0, step=0.1)
        caffeine     = st.slider("Caffeine cups/day", 0, 8, 1)
    with c2:
        phone_hours  = st.slider("Smartphone hrs/day", 0.0, 14.0, 5.5, step=0.1)
        social_media = st.slider("Social media hrs/day", 0.0, 9.0, 3.5, step=0.1)
        gaming       = st.slider("Gaming hrs/day", 0.0, 8.0, 2.2, step=0.1)
        streaming    = st.slider("Streaming hrs/day", 0.0, 6.0, 1.8, step=0.1)
        attendance   = st.slider("Class attendance %", 45, 100, 85)
        assignment   = st.slider("Assignment completion %", 30, 100, 80)

    c3,c4 = st.columns(2)
    with c3:
        gender        = st.selectbox("Gender", ["Female","Male"])
        mental_health = st.selectbox("Mental health", ["Good","Average","Poor"])
    with c4:
        parent_edu    = st.selectbox("Parent education", ["PhD","Masters","Bachelors","HighSchool"])
        internet      = st.selectbox("Internet quality", ["Good","Average","Poor"])

    total_screen    = phone_hours + social_media + gaming + streaming
    study_eff       = study_hours / (total_screen + 1)
    academic_commit = (attendance + assignment) / 2
    wellbeing       = sleep_hours + exercise

    input_df = pd.DataFrame([{
        'age': age, 'study_hours_per_day': study_hours,
        'smartphone_usage_hours': phone_hours, 'social_media_hours': social_media,
        'gaming_hours': gaming, 'streaming_hours': streaming,
        'sleep_hours': sleep_hours, 'exercise_hours': exercise,
        'class_attendance_percent': attendance, 'assignment_completion_percent': assignment,
        'caffeine_intake_cups': caffeine, 'motivation_level': motivation,
        'gender': 0 if gender=="Female" else 1,
        'mental_health_status': {"Good":1,"Average":0,"Poor":2}[mental_health],
        'parent_education_level': {"PhD":3,"Masters":2,"Bachelors":0,"HighSchool":1}[parent_edu],
        'internet_quality': {"Good":1,"Average":0,"Poor":2}[internet],
        'total_screen_time': total_screen, 'study_efficiency': study_eff,
        'academic_commitment': academic_commit, 'wellbeing_score': wellbeing
    }])

    st.markdown("---")
    if st.button("🔮 Predict My Exam Score", use_container_width=True, type="primary"):
        score = float(np.clip(md['gb'].predict(input_df)[0], 0, 100))
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Predicted Score",  f"{score:.1f} / 100")
        m2.metric("Screen Time/Day",  f"{total_screen:.1f} hrs")
        m3.metric("Study Efficiency", f"{study_eff:.2f}")
        m4.metric("Wellbeing Score",  f"{wellbeing:.1f}")

        if   score >= 88: st.success("🌟 Excellent! You're in the top tier. Keep it up!")
        elif score >= 75: st.info("👍 Good performance! Slightly more study hours → Excellent.")
        elif score >= 60: st.warning("⚠️ Average. Reduce screen time and increase study hours.")
        else:             st.error("🚨 At Risk. Focus on attendance and consistent daily study.")

        st.markdown("---")
        st.markdown("#### Your habits vs dataset average")
        cats     = ['Study hrs','Sleep hrs','Attendance %','Assignment %','Motivation','Screen time']
        avg_vals = [df['study_hours_per_day'].mean(), df['sleep_hours'].mean(),
                    df['class_attendance_percent'].mean(), df['assignment_completion_percent'].mean(),
                    df['motivation_level'].mean(),
                    (df['smartphone_usage_hours']+df['social_media_hours']+df['gaming_hours']+df['streaming_hours']).mean()]
        user_vals = [study_hours, sleep_hours, attendance, assignment, motivation, total_screen]
        x = np.arange(len(cats))
        fig,ax = plt.subplots(figsize=(10,4))
        ax.bar(x-0.18, avg_vals,  0.35, label='Dataset avg', color='#7F77DD', alpha=0.75, edgecolor='black', linewidth=0.4)
        ax.bar(x+0.18, user_vals, 0.35, label='Your values', color='#2ECC71', alpha=0.9,  edgecolor='black', linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
        ax.legend(); ax.set_ylabel("Value"); clean(ax)
        st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:13px; padding:8px 0'>
    Built by <b>Saurabh Anand</b> &nbsp;|&nbsp;
    <a href='https://github.com/SaurabhAnand56' target='_blank' style='color:#aaa'>GitHub</a> &nbsp;|&nbsp;
    <a href='https://www.linkedin.com/in/saurabhanand56' target='_blank' style='color:#aaa'>LinkedIn</a> &nbsp;|&nbsp;
    Dataset: 15,000 students &nbsp;|&nbsp; Model: Gradient Boosting R²=0.68
</div>
""", unsafe_allow_html=True)
