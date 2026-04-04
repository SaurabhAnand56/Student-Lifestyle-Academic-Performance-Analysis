import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='Student Score Predictor', page_icon='🎓')
st.title('🎓 Student Exam Score Predictor')
st.caption('Enter your lifestyle habits to predict your exam score')

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider('Study hours per day', 0.0, 12.0, 4.5)
    sleep_hours = st.slider('Sleep hours per day', 3.5, 10.0, 7.0)
    motivation  = st.slider('Motivation level (1-10)', 1.0, 10.0, 6.0)
    attendance  = st.slider('Class attendance %', 45, 100, 85)

with col2:
    phone_hours   = st.slider('Smartphone usage (hrs/day)', 0.0, 14.0, 5.5)
    assignment_pct= st.slider('Assignment completion %', 30, 100, 80)
    mental_health = st.selectbox('Mental health status', ['Good','Average','Poor'])
    internet      = st.selectbox('Internet quality', ['Good','Average','Poor'])

if st.button('Predict My Score 🔮'):
    # Simple scoring formula based on feature importances
    score = (study_hours * 8.5) + (sleep_hours * 0.8) + (motivation * 1.2) \
          + (attendance * 0.15) + (assignment_pct * 0.12) - (phone_hours * 1.1)
    score = np.clip(score, 0, 100)
    
    st.metric('Predicted Exam Score', f'{score:.1f} / 100')
    
    if score >= 88:
        st.success('Excellent! Keep it up 🌟')
    elif score >= 75:
        st.info('Good performance! Small improvements can push you higher.')
    elif score >= 60:
        st.warning('Average range. Try increasing study hours and reducing screen time.')
    else:
        st.error('At Risk. Consider speaking with your academic advisor.')
```