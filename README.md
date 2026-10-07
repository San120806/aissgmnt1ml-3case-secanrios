# Machine Learning Fundamentals — Assignment 1 & Practical Project

This repository contains the complete answers, practical implementation codebase, interactive dashboard, and a professional 3-page PDF report for **Assignment 1: Machine Learning Fundamentals (10 Marks)**.

---

## 📁 Repository Structure

```text
mlassignmnts/
├── ML_Fundamentals_Assignment_Report.pdf  # 📄 Beautiful 3-page submission PDF report
├── ASSIGNMENT_ANSWERS.md                 # 📝 Direct Markdown copy of all answers
├── ml_fundamentals_assignment.ipynb      # 📓 Executed Jupyter Notebook with charts & notes
├── ml_fundamentals_project.py           # 🐍 Standalone Python training & evaluation script
├── app.py                               # 🌐 Interactive Streamlit web demo
├── figures/                             # 📊 Generated plots & visual charts
│   ├── scenario_1_spend_regression.png
│   ├── scenario_2_fraud_confusion_matrix.png
│   ├── scenario_3_user_clusters.png
│   └── part_b_student_pass_fail.png
└── generate_pdf_report.py               # 🖨️ ReportLab script to regenerate the PDF
```

---

## 🚀 How to Run the Project

### 1. Run the Python Models & Regenerate Plots
```bash
python3 ml_fundamentals_project.py
```

### 2. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
*This allows you to adjust sliders (e.g. past spend, transaction amount, attendance rate) and see live predictions from each model.*

### 3. Open the Jupyter Notebook
```bash
jupyter notebook ml_fundamentals_assignment.ipynb
```

### 4. Regenerate the Assignment PDF Report
```bash
python3 generate_pdf_report.py
```

---

## 📌 Deliverables Summary at a Glance

| Scenario | (a) Paradigm | (b) Task Type | (c) 2–3 Likely Features | (d) The Target |
|:---|:---|:---|:---|:---|
| **1. Customer Spend** | Supervised | Regression | Past spend, Frequency, Recency | `next_month_spend` ($) |
| **2. Fraud Detection** | Supervised | Binary Classification | Amount, Distance, Velocity | `is_fraud` (0 or 1) |
| **3. Viewing Habits** | Unsupervised | Clustering | Watch hours, Weekend ratio, Genres | *None (Discovers Groups)* |
| **4. Student Exam (Part B)** | Supervised | Binary Classification | Attendance %, Study hrs, Assignments | `course_passed` (0 or 1) |

- **Part C Risk Analysis:** Evaluated on **Scenario 2 (Fraud Detection)** focusing on **class imbalance (99% vs 1%)**, **asymmetric cost of false positives (customer churn) vs false negatives (chargebacks)**, and **adversarial drift** (Word count: ~255 words).
