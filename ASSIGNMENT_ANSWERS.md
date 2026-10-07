# Machine Learning Fundamentals — Assignment 1

**Course:** Introduction to Machine Learning  
**Assignment:** Question 1 (10 Marks)  
**Deliverables:** Conceptual Answers, Applied Scenarios, and Risk Analysis

---

## Part A: Scenario Analysis (Scenarios 1 – 3)

For each scenario below, the deliverables state:
- **(a)** Supervised or Unsupervised
- **(b)** Classification or Regression
- **(c)** 2–3 Likely Features
- **(d)** The Target

---

### 1. Predict how much a customer will spend next month from their past purchases.

- **(a) Supervised or Unsupervised:**  
  **Supervised Learning**  
  *Rationale:* Historical transactional databases contain past purchasing logs paired with the confirmed dollar amount spent in the following month (ground-truth training labels).

- **(b) Classification or Regression:**  
  **Regression**  
  *Rationale:* The predicted quantity is a continuous numerical dollar amount (e.g., `$145.50`), rather than a discrete category or class.

- **(c) 2–3 Likely Features:**  
  1. `past_avg_spend`: Average dollar amount spent per month over the past 3 to 6 months (reflects purchasing capacity).
  2. `purchase_frequency`: Number of distinct orders or transactions per month (reflects habitual engagement).
  3. `recency_days`: Days elapsed since the customer's last purchase (distinguishes active loyal buyers from lapsing customers).

- **(d) The Target:**  
  `next_month_spend` — Continuous numerical target variable in currency ($).

---

### 2. Flag whether a transaction is fraud or not.

- **(a) Supervised or Unsupervised:**  
  **Supervised Learning**  
  *Rationale:* Supervised models train on historical credit card transaction histories where past transactions have confirmed labels (1 = Fraudulent, 0 = Legitimate), confirmed by cardholders or fraud investigation operations.

- **(b) Classification or Regression:**  
  **Classification (Binary Classification)**  
  *Rationale:* The output belongs to one of two discrete, mutually exclusive classes: `Fraud` (1) or `Legitimate` (0).

- **(c) 2–3 Likely Features:**  
  1. `transaction_amount`: Monetary charge value (unusually large amounts deviate from historical baselines).
  2. `distance_from_home_km`: Physical distance between cardholder's usual location/IP and the purchase terminal.
  3. `transaction_velocity_hour`: Rapid charge attempts within a short window or unusual hours (e.g., 3:30 AM).

- **(d) The Target:**  
  `is_fraud` — Binary categorical target variable (`1` for Fraudulent, `0` for Legitimate).

---

### 3. Group users into segments based on viewing habits (no preset labels).

- **(a) Supervised or Unsupervised:**  
  **Unsupervised Learning**  
  *Rationale:* The prompt specifically notes that there are *"no preset labels"*. Without ground-truth answers or a teacher, the algorithm must independently discover patterns and natural groupings.

- **(b) Classification or Regression:**  
  **Neither (Clustering)**  
  *Rationale:* Classification requires predefined classes. This task organizes data points based on geometric similarity/proximity into organic cohorts (e.g., using K-Means or DBSCAN).

- **(c) 2–3 Likely Features:**  
  1. `weekly_watch_hours`: Total hours spent streaming content per week.
  2. `weekend_viewing_ratio`: Ratio of watch time occurring on weekends (Sat–Sun) versus weekdays (Mon–Fri).
  3. `genre_diversity_ratio`: Proportion of watch time distributed across diverse content categories (e.g., movies, series, sports).

- **(d) The Target:**  
  **None (No target variable)**  
  *Rationale:* Unsupervised learning problems have no predefined ground-truth label. The algorithm outputs an assigned cluster identifier (e.g., Cluster 0, Cluster 1, Cluster 2).

---

## Part B: Student-Created Scenario

### Scenario:
> **"Predict whether a university student will pass or fail a course based on semester study habits."**

- **(a) Supervised or Unsupervised:**  
  **Supervised Learning**  
  *Rationale:* The institution holds historical academic records from previous cohorts containing students' semester study behaviors alongside their confirmed final course outcome (Passed vs Failed).

- **(b) Classification or Regression:**  
  **Classification (Binary Classification)**  
  *Rationale:* The outcome variable is a categorical binary state: Pass (1) or Fail (0).

- **(c) 2–3 Likely Features:**  
  1. `lecture_attendance_pct`: Percentage of scheduled lectures and laboratory sessions attended.
  2. `weekly_self_study_hours`: Number of hours devoted to independent study and revision each week.
  3. `assignment_completion_rate`: Percentage of required assignments submitted on time.

- **(d) The Target:**  
  `course_passed` — Binary categorical target variable (`1` = Pass, `0` = Fail).

---

## Part C: Critical Risk Analysis of Misapplying ML

**Selected Scenario:** **Scenario 2 — Fraudulent Transaction Detection**  
**Word Count:** **255 Words** *(Within recommended 200–300 words range)*

A major risk of misapplying machine learning to fraud detection is failing to account for **severe class imbalance** and the resulting **asymmetric cost of prediction errors**. In real-world payment networks, legitimate transactions make up over 99% of all activity, while actual fraud represents less than 1%. If a practitioner naively optimizes for overall accuracy, a completely trivial model that predicts 'legitimate' 100% of the time achieves 99% accuracy while catching zero fraudulent transactions, rendering the system practically useless.

Furthermore, improper thresholding produces severe business consequences. If decision boundaries are tuned too aggressively to catch every potential threat, the rate of **False Positives** spikes. When honest cardholders traveling abroad or making atypical anniversary purchases have their cards blocked, it causes acute customer frustration. Industry studies indicate that insulted customers frequently abandon their payment cards and switch to competitors, inflicting long-term loss of customer lifetime value that often exceeds the cost of fraud itself. Conversely, excessive **False Negatives** cause direct financial liability, chargeback fees, and regulatory penalties.

Lastly, deploying supervised models as static black boxes overlooks **adversarial concept drift**. Organized fraud syndicates rapidly adapt patterns to bypass fixed thresholds. Without continuous retraining, cost-sensitive loss calibration, precision-recall curve monitoring, and human-in-the-loop dispute workflows, institutions acquire a false sense of automation while remaining highly exposed.

---

## Summary Comparison Matrix

| Scenario | Paradigm (a) | Task Type (b) | Key Features (c) | Target Variable (d) |
|---|---|---|---|---|
| **1. Customer Spend** | Supervised | Regression | Past spend, Frequency, Recency | `next_month_spend` ($) |
| **2. Fraud Detection** | Supervised | Binary Classification | Amount, Distance, Velocity | `is_fraud` (0 or 1) |
| **3. Viewing Habits** | Unsupervised | Clustering | Watch hours, Weekend ratio, Genres | *None (Discovers Groups)* |
| **4. Student Exam (Part B)** | Supervised | Binary Classification | Attendance %, Study hrs, Assignments | `course_passed` (0 or 1) |
