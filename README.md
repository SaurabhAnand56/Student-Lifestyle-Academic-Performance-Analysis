# 📊 Student Lifestyle & Academic Performance

This is my **first EDA (Exploratory Data Analysis) project**.  
I analyzed data of 15,000 students to find out what habits affect their exam scores.

> 🙋 I was stuck in tutorial hell for a long time. This is my first step out of it — a real project with real data.

---

## 📊 Power BI Dashboard

### Overview
![Dashboard Overview](images/Academic-Performance-Overview.png)

### Deep Dive
![Dashboard Detail](images/Student-Lifestyle-Insights.png)

---

## 🤔 What Questions I Tried to Answer

- Does studying more actually improve scores?
- Does using social media or gaming hurt performance?
- Does mental health affect grades?
- What do top-performing students have in common?

---

## 📁 About the Dataset

The dataset has **15,000 rows** and **18 columns**.  
Each row = one student. Each column = one detail about that student.

| Column | What it means |
|--------|---------------|
| `study_hours_per_day` | How many hours the student studies daily |
| `smartphone_usage_hours` | Daily phone usage |
| `social_media_hours` | Time spent on social media |
| `sleep_hours` | How many hours they sleep |
| `class_attendance_percent` | What % of classes they attended |
| `mental_health_status` | Good / Average / Poor |
| `final_exam_score` | Their exam score (this is what I'm analyzing) |

✅ No missing values — the data was already clean.

---

## 🛠️ Tools I Used

| Tool | Why I used it |
|------|---------------|
| Python | Main programming language |
| Pandas | To load and work with the data |
| Seaborn & Matplotlib | To make charts |
| Jupyter Notebook | To write code and notes together |
| Power BI | To make an interactive dashboard |

---

## 🔍 What I Did (Step by Step)

### 1. Loaded the data
```python
df = pd.read_csv("student_digital_life.csv")
df.head()  # shows first 5 rows
```

### 2. Checked for problems
```python
df.isnull().sum()     # missing values? → None ✅
df.duplicated().sum() # duplicate rows? → None ✅
```

### 3. Created a new column
```python
df["total_screen_time"] = (
    df["social_media_hours"] +
    df["gaming_hours"] +
    df["streaming_hours"]
)
```
Combined all screen time into one column — this is called **feature engineering**.

### 4. Explored each column (Univariate Analysis)
Made charts for individual columns to understand distributions.

- Most students study **3–6 hours/day**
- Many students use their phone **5+ hours/day**
- Age is mostly between **18–23**

### 5. Compared columns to exam scores (Bivariate Analysis)
Made scatter plots to see relationships.

- More study hours → Higher exam scores ✅
- More attendance → Higher scores ✅
- More screen time → Slightly lower scores 📉

### 6. Looked at groups (Categorical Analysis)
Used box plots to compare scores across groups.

- **Mental health matters** — Good = avg 84, Poor = avg 73
- **Gender doesn't matter** — scores are nearly the same

### 7. Correlation Heatmap
Shows which columns are most related to exam scores.

**Top factors affecting exam score:**
- Study hours
- Class attendance
- Assignment completion

---

## 💡 What I Found

1. **Study hours are the #1 factor** — students who study more score higher
2. **Attendance really matters** — skipping class hurts your grade
3. **Mental health affects grades** — poor mental health = ~11 points lower on average
4. **Gender has no effect** on exam scores in this dataset
5. **Too much screen time** has a small negative effect on scores

---

## 😅 Honest Note

I spent way too long watching tutorials and not building anything.  
This project is me finally doing something real — it's not perfect, but it's mine.  
If you're also stuck in tutorial hell, just start. The learning happens when you build.

---

## 📬 Let's Connect

[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/SaurabhAnand56)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://linkedin.com/in/saurabhanand56)
