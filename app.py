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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Digital Life Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.insight-box {
    background: #1a1a2e;
    border-left: 4px solid #2ECC71;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dataset/student_digital_life.csv')

# ── Train all models once ─────────────────────────────────────────────────────
@st.cache_resource
def train_all(_df):
    df_ml = _df.copy()
    le = LabelEncoder()
    for col in ['gender','mental_health_status','parent_education_level','internet_quality']:
        df_ml[col] = le.fit_transform(df_ml[col])

    df_ml['total_screen_time']   = df_ml['smartphone_usage_hours'] + df_ml['social_media_hours'] + df_ml['gaming_hours'] + df_ml['streaming_hours']
    df_ml['study_efficiency']    = df_ml['study_hours_per_day'] / (df_ml['total_screen_time'] + 1)
    df_ml['academic_commitment'] = (df_ml['class_attendance_percent'] + df_ml['assignment_completion_percent']) / 2
    df_ml['wellbeing_score']     = df_ml['sleep_hours'] + df_ml['exercise_hours']
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

    X     = df_ml[features]
    y_reg = df_ml['final_exam_score']
    y_clf = df_ml['performance_category']

    Xtr,Xte,ytr,yte         = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    Xtr_c,Xte_c,ytr_c,yte_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    gb  = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
    rf  = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
    lr  = LinearRegression().fit(Xtr, ytr)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(Xtr_c, ytr_c)

    scaler   = StandardScaler()
    Xs       = scaler.fit_transform(X)
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_ml['cluster'] = km.fit_predict(Xs)

    return dict(df_ml=df_ml, features=features,
                Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                Xte_c=Xte_c, yte_c=yte_c,
                gb=gb, rf=rf, lr=lr, clf=clf,
                Xs=Xs, scaler=scaler)

df = load_data()
md = train_all(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://github.com/SaurabhAnand56.png", width=90)
    st.markdown("### Saurabh Anand")
    st.markdown("Data Analyst | Python • Power BI • SQL")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github)](https://github.com/SaurabhAnand56)  "
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin)](https://www.linkedin.com/in/saurabhanand56)"
    )
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "📊  EDA — Overview",
        "🔗  EDA — Relationships",
        "🤖  ML — Regression",
        "🏷️  ML — Classification",
        "🔍  ML — Feature Importance",
        "🗂️  ML — Clustering",
        "🔮  Predict My Score"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} students | 18 features")
    st.caption("Best model: Gradient Boosting R²=0.68")

# ══════════════════════════════════════════
# HOME
# ══════════════════════════════════════════
if page == "🏠  Home":
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image("https://github.com/SaurabhAnand56.png", width=110)
    with c2:
        st.title("Student Digital Life Analysis")
        st.markdown("**Author:** Saurabh Anand")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-SaurabhAnand56-181717?logo=github&style=flat)](https://github.com/SaurabhAnand56)  "
            "[![LinkedIn](https://img.shields.io/badge/LinkedIn-saurabhanand56-0A66C2?logo=linkedin&style=flat)](https://www.linkedin.com/in/saurabhanand56)"
        )
    st.markdown("---")
    st.markdown("""
    This project explores how **digital habits** — screen time, gaming, sleep, study hours —
    affect students' **final exam scores**. Dataset: **15,000 students, 18 features**.

    **What's inside:**
    - Full Exploratory Data Analysis (EDA)
    - Regression, Classification & Clustering ML models
    - Live score prediction from your own habits
    """)
    st.markdown("### Dataset snapshot")
    a,b,c,d = st.columns(4)
    a.metric("Students",       f"{len(df):,}")
    b.metric("Avg Exam Score", f"{df['final_exam_score'].mean():.1f}")
    c.metric("Avg Study Hrs",  f"{df['study_hours_per_day'].mean():.1f}/day")
    d.metric("Avg Screen Time",f"{(df['smartphone_usage_hours']+df['social_media_hours']+df['gaming_hours']+df['streaming_hours']).mean():.1f} hrs/day")
    st.markdown("### Key findings")
    st.markdown('<div class="insight-box">📚 <b>Study hours</b> — #1 predictor (49% feature importance, r=0.64)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">📱 <b>Smartphone usage</b> — top negative factor (r=−0.14)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">🤖 <b>Gradient Boosting</b> — best model: R²=0.68, RMSE=10.59</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">👥 <b>K-Means</b> — 4 distinct student behaviour profiles found</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Sample data")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════
# EDA OVERVIEW
# ══════════════════════════════════════════
elif page == "📊  EDA — Overview":
    st.title("📊 EDA — Overview")
    st.markdown("Distribution of individual variables across 15,000 students.")
    st.markdown("---")

    def clean_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("#### Age distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['age'], bins=9, color='#7F77DD', edgecolor='black', linewidth=0.4, ax=ax)
        ax.set_xlabel("Age"); clean_ax(ax)
        st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("#### Gender distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        cnt = df['gender'].value_counts()
        ax.bar(cnt.index, cnt.values, color=['#E57373','#64B5F6'], edgecolor='black', linewidth=0.4)
        for i,v in enumerate(cnt.values): ax.text(i, v+50, f'{v:,}', ha='center', fontweight='bold')
        clean_ax(ax)
        st.pyplot(fig); plt.close()

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("#### Mental health status")
        fig, ax = plt.subplots(figsize=(6,4))
        cnt = df['mental_health_status'].value_counts()
        ax.bar(cnt.index, cnt.values, color=['#2ECC71','#F39C12','#E74C3C'], edgecolor='black', linewidth=0.4)
        for i,v in enumerate(cnt.values): ax.text(i, v+50, f'{v:,}', ha='center', fontweight='bold')
        clean_ax(ax)
        st.pyplot(fig); plt.close()
        st.info("Majority report Average or Good mental health.")

    with r2c2:
        st.markdown("#### Study hours per day")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['study_hours_per_day'], bins=25, color='#2ECC71', edgecolor='black', linewidth=0.3, ax=ax)
        ax.axvline(df['study_hours_per_day'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['study_hours_per_day'].mean():.1f} hrs")
        ax.legend(); ax.set_xlabel("Study Hours"); clean_ax(ax)
        st.pyplot(fig); plt.close()

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.markdown("#### Smartphone usage per day")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['smartphone_usage_hours'], bins=25, color='#E74C3C', edgecolor='black', linewidth=0.3, ax=ax)
        ax.axvline(df['smartphone_usage_hours'].mean(), color='blue', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['smartphone_usage_hours'].mean():.1f} hrs")
        ax.legend(); ax.set_xlabel("Phone Hours"); clean_ax(ax)
        st.pyplot(fig); plt.close()

    with r3c2:
        st.markdown("#### Final exam score distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['final_exam_score'], bins=30, color='#F39C12', edgecolor='black', linewidth=0.3, ax=ax)
        ax.axvline(df['final_exam_score'].mean(),   color='red',  linestyle='--', linewidth=1.5, label=f"Mean: {df['final_exam_score'].mean():.1f}")
        ax.axvline(df['final_exam_score'].median(), color='blue', linestyle='--', linewidth=1.5, label=f"Median: {df['final_exam_score'].median():.1f}")
        ax.legend(); ax.set_xlabel("Exam Score"); clean_ax(ax)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Full correlation heatmap")
    fig, ax = plt.subplots(figsize=(13,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.4, ax=ax, annot_kws={'size':8})
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
# EDA RELATIONSHIPS
# ══════════════════════════════════════════
elif page == "🔗  EDA — Relationships":
    st.title("🔗 EDA — Relationships with Exam Score")
    st.markdown("---")

    def scatter_with_trendline(col, color, ax):
        ax.scatter(df[col], df['final_exam_score'], alpha=0.15, s=6, color=color)
        m,b = np.polyfit(df[col], df['final_exam_score'], 1)
        xl = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(xl, m*xl+b, 'r-', linewidth=2)
        ax.set_xlabel(col); ax.set_ylabel("Exam Score")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Study hours vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_with_trendline('study_hours_per_day','#7F77DD', ax)
        st.pyplot(fig); plt.close()
        st.success("Strong positive correlation  r = 0.64")

    with c2:
        st.markdown("#### Smartphone usage vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_with_trendline('smartphone_usage_hours','#E74C3C', ax)
        st.pyplot(fig); plt.close()
        st.error("Negative correlation  r = −0.14")

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("#### Class attendance vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_with_trendline('class_attendance_percent','#2ECC71', ax)
        st.pyplot(fig); plt.close()
        st.success("Positive correlation  r = 0.14")

    with c4:
        st.markdown("#### Sleep hours vs exam score")
        fig,ax = plt.subplots(figsize=(6,4))
        scatter_with_trendline('sleep_hours','#3498DB', ax)
        st.pyplot(fig); plt.close()
        st.info("Modest positive  r = 0.07")

    c5,c6 = st.columns(2)
    with c5:
        st.markdown("#### Exam score by gender")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='gender', y='final_exam_score', data=df, palette=['#E57373','#64B5F6'], ax=ax)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with c6:
        st.markdown("#### Exam score by mental health")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='mental_health_status', y='final_exam_score', data=df,
                    order=['Good','Average','Poor'], palette=['#2ECC71','#F39C12','#E74C3C'], ax=ax)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()
        st.info("Good mental health → higher scores on average.")

# ══════════════════════════════════════════
# ML REGRESSION
# ══════════════════════════════════════════
elif page == "🤖  ML — Regression":
    st.title("🤖 ML — Exam Score Regression")
    st.markdown("Comparing Linear Regression, Random Forest, and Gradient Boosting.")
    st.markdown("---")

    gb_p = md['gb'].predict(md['Xte'])
    rf_p = md['rf'].predict(md['Xte'])
    lr_p = md['lr'].predict(md['Xte'])

    results = pd.DataFrame({
        'Model':   ['Linear Regression','Random Forest','Gradient Boosting'],
        'R²':      [round(r2_score(md['yte'],lr_p),4), round(r2_score(md['yte'],rf_p),4), round(r2_score(md['yte'],gb_p),4)],
        'RMSE':    [round(np.sqrt(mean_squared_error(md['yte'],lr_p)),2),
                    round(np.sqrt(mean_squared_error(md['yte'],rf_p)),2),
                    round(np.sqrt(mean_squared_error(md['yte'],gb_p)),2)]
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Actual vs Predicted (Gradient Boosting)")
        fig,ax = plt.subplots(figsize=(6,5))
        ax.scatter(md['yte'], gb_p, alpha=0.25, color='steelblue', s=8)
        ax.plot([md['yte'].min(),md['yte'].max()],[md['yte'].min(),md['yte'].max()],'r--',linewidth=1.5,label='Perfect fit')
        ax.text(5,95,'R² = 0.68',fontsize=12,color='red',fontweight='bold')
        ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
        ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### R² Comparison")
        fig,ax = plt.subplots(figsize=(6,5))
        bars = ax.bar(results['Model'], results['R²'], color=['#3498db','#2ecc71','#e74c3c'],
                      edgecolor='black', linewidth=0.5)
        ax.set_ylim(0,1); ax.set_ylabel("R² Score")
        for bar,val in zip(bars,results['R²']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, str(val), ha='center', fontweight='bold')
        plt.xticks(rotation=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Residual plot")
    residuals = md['yte'] - gb_p
    fig,ax = plt.subplots(figsize=(12,4))
    ax.scatter(gb_p, residuals, alpha=0.2, color='#7F77DD', s=8)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Predicted Score"); ax.set_ylabel("Residual")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig); plt.close()
    st.info("Residuals scattered randomly around 0 — no major systematic bias.")

# ══════════════════════════════════════════
# ML CLASSIFICATION
# ══════════════════════════════════════════
elif page == "🏷️  ML — Classification":
    st.title("🏷️ ML — Performance Classifier")
    st.markdown("Random Forest classifying students: At Risk / Average / Good / Excellent")
    st.markdown("---")

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
        ax.bar(cnt.index, cnt.values, color=['#e74c3c','#f39c12','#3498db','#2ecc71'],
               edgecolor='black', linewidth=0.5)
        for i,v in enumerate(cnt.values): ax.text(i, v+30, f'{v:,}', ha='center', fontweight='bold', fontsize=9)
        ax.set_ylabel("Students")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Confusion matrix")
        fig,ax = plt.subplots(figsize=(6,5))
        cm = confusion_matrix(md['yte_c'], y_pred_c, labels=['At Risk','Average','Good','Excellent'])
        ConfusionMatrixDisplay(cm, display_labels=['At Risk','Average','Good','Excellent']).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Classification report")
    report_df = pd.DataFrame(classification_report(md['yte_c'], y_pred_c, output_dict=True)).T.round(2)
    st.dataframe(report_df, use_container_width=True)
    st.info("Best at identifying Excellent and At-Risk students. Average vs Good is hardest to separate — they share very similar habits.")

# ══════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════
elif page == "🔍  ML — Feature Importance":
    st.title("🔍 ML — Feature Importance")
    st.markdown("Which habits matter most for predicting exam scores?")
    st.markdown("---")

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
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Pearson correlation with exam score")
    num_cols = ['study_hours_per_day','smartphone_usage_hours','social_media_hours','gaming_hours',
                'sleep_hours','exercise_hours','motivation_level','class_attendance_percent',
                'assignment_completion_percent','final_exam_score']
    corr = df[num_cols].corr()['final_exam_score'].drop('final_exam_score').sort_values()
    colors_corr = ['#E74C3C' if x < 0 else '#2ECC71' for x in corr]
    fig,ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(corr.index, corr.values, color=colors_corr, edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Pearson r")
    for bar,val in zip(bars,corr.values):
        ax.text(val+(0.005 if val>=0 else -0.005), bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', ha='left' if val>=0 else 'right', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.success("study_hours_per_day leads in both correlation (r=0.64) and feature importance (49%).")

# ══════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════
elif page == "🗂️  ML — Clustering":
    st.title("🗂️ ML — Student Segmentation (K-Means)")
    st.markdown("4 distinct student behaviour profiles identified from 15,000 records.")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Elbow method")
        inertias = []
        for k in range(2,11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(md['Xs'])
            inertias.append(km.inertia_)
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(range(2,11), inertias, 'o-', color='steelblue', linewidth=2, markersize=7)
        ax.axvline(4, color='red', linestyle='--', linewidth=1.5, label='K=4')
        ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.legend()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Exam score by cluster")
        fig,ax = plt.subplots(figsize=(6,4))
        data = [md['df_ml'][md['df_ml']['cluster']==i]['final_exam_score'].values for i in range(4)]
        bp = ax.boxplot(data, patch_artist=True, labels=[f'Cluster {i}' for i in range(4)])
        for patch,col in zip(bp['boxes'],['#3498db','#2ecc71','#e74c3c','#f39c12']):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_ylabel("Exam Score")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
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

    st.markdown("---")
    st.markdown("#### PCA 2D cluster visualisation")
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(md['Xs'])
    fig,ax = plt.subplots(figsize=(10,5))
    sc = ax.scatter(coords[:,0], coords[:,1], c=md['df_ml']['cluster'], cmap='tab10', alpha=0.35, s=6)
    plt.colorbar(sc, ax=ax, label='Cluster')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
# PREDICT MY SCORE
# ══════════════════════════════════════════
elif page == "🔮  Predict My Score":
    st.title("🔮 Predict My Exam Score")
    st.markdown("Live prediction using Gradient Boosting (R²=0.68, RMSE=10.59)")
    st.markdown("---")

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
    if st.button("Predict My Exam Score", use_container_width=True, type="primary"):
        score = float(np.clip(md['gb'].predict(input_df)[0], 0, 100))
        m1,m2,m3 = st.columns(3)
        m1.metric("Predicted Score",  f"{score:.1f} / 100")
        m2.metric("Screen Time/Day",  f"{total_screen:.1f} hrs")
        m3.metric("Study Efficiency", f"{study_eff:.2f}")

        if   score >= 88: st.success("Excellent! You're in the top tier. Keep it up!")
        elif score >= 75: st.info("Good! A small push in study hours gets you to Excellent.")
        elif score >= 60: st.warning("Average. Reduce screen time and increase study hours.")
        else:             st.error("At Risk. Focus on attendance and consistent study.")

        st.markdown("---")
        st.markdown("#### You vs dataset average")
        cats      = ['Study hrs','Sleep hrs','Attendance %','Assignment %','Motivation','Screen time']
        avg_vals  = [df['study_hours_per_day'].mean(), df['sleep_hours'].mean(),
                     df['class_attendance_percent'].mean(), df['assignment_completion_percent'].mean(),
                     df['motivation_level'].mean(),
                     (df['smartphone_usage_hours']+df['social_media_hours']+df['gaming_hours']+df['streaming_hours']).mean()]
        user_vals = [study_hours, sleep_hours, attendance, assignment, motivation, total_screen]
        x = np.arange(len(cats))
        fig,ax = plt.subplots(figsize=(10,4))
        ax.bar(x-0.18, avg_vals,  0.35, label='Dataset avg', color='#7F77DD', alpha=0.75, edgecolor='black', linewidth=0.4)
        ax.bar(x+0.18, user_vals, 0.35, label='Your values', color='#2ECC71', alpha=0.9,  edgecolor='black', linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
        ax.legend(); ax.set_ylabel("Value")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()
