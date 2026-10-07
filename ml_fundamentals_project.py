"""
Machine Learning Fundamentals - Assignment 1 Project
Written from the perspective of a beginner ML student.

This script demonstrates:
1. Scenario 1: Customer Spend Prediction (Supervised Regression)
2. Scenario 2: Fraudulent Transaction Detection (Supervised Classification)
3. Scenario 3: Viewing Habit Segmentation (Unsupervised Clustering)
4. Part B: Student Exam Pass/Fail Prediction (Supervised Classification)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_scenario_1():
    print("=" * 60)
    print("SCENARIO 1: Predict Customer Spend Next Month")
    print("Type: Supervised Learning | Regression")
    print("Features: past_avg_spend, purchase_frequency, days_since_last_purchase")
    print("Target: next_month_spend ($)")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 300
    past_avg_spend = np.random.uniform(20, 500, n_samples)
    purchase_frequency = np.random.poisson(4, n_samples) + 1
    days_since_last_purchase = np.random.uniform(1, 30, n_samples)

    # Next month spend roughly depends on past average spend and frequency, inversely with recency
    noise = np.random.normal(0, 25, n_samples)
    next_month_spend = (0.85 * past_avg_spend) + (12 * purchase_frequency) - (1.2 * days_since_last_purchase) + noise
    next_month_spend = np.maximum(next_month_spend, 5.0)  # non-negative

    df = pd.DataFrame({
        'past_avg_spend': past_avg_spend,
        'purchase_frequency': purchase_frequency,
        'days_since_last_purchase': days_since_last_purchase,
        'next_month_spend': next_month_spend
    })

    X = df[['past_avg_spend', 'purchase_frequency', 'days_since_last_purchase']]
    y = df['next_month_spend']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: ${rmse:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.scatter(y_test, y_pred, color='#2563eb', alpha=0.7, edgecolors='none', label='Test Predictions')
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', label='Perfect 1:1 Fit')
    ax.set_title("Scenario 1: Actual vs Predicted Customer Spend ($)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Actual Next Month Spend ($)")
    ax.set_ylabel("Predicted Next Month Spend ($)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "scenario_1_spend_regression.png")
    fig.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}\n")


def run_scenario_2():
    print("=" * 60)
    print("SCENARIO 2: Flag Whether a Transaction is Fraud or Not")
    print("Type: Supervised Learning | Binary Classification")
    print("Features: amount, distance_from_home_km, hour_of_day")
    print("Target: is_fraud (0 = Legitimate, 1 = Fraud)")
    print("=" * 60)

    np.random.seed(42)
    n_legit = 900
    n_fraud = 100

    # Legitimate transactions: lower amount, closer to home
    legit_amount = np.random.exponential(scale=45, size=n_legit) + 5
    legit_dist = np.random.exponential(scale=10, size=n_legit)
    legit_target = np.zeros(n_legit)

    # Fraudulent transactions: unusually high amount or unusual distance
    fraud_amount = np.random.exponential(scale=250, size=n_fraud) + 80
    fraud_dist = np.random.exponential(scale=85, size=n_fraud) + 20
    fraud_target = np.ones(n_fraud)

    amounts = np.concatenate([legit_amount, fraud_amount])
    distances = np.concatenate([legit_dist, fraud_dist])
    targets = np.concatenate([legit_target, fraud_target])

    df = pd.DataFrame({
        'amount': amounts,
        'distance_km': distances,
        'is_fraud': targets
    })

    X = df[['amount', 'distance_km']]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print("Confusion Matrix:\n", cm)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Legit (0)', 'Fraud (1)'], yticklabels=['Legit (0)', 'Fraud (1)'])
    ax.set_title("Scenario 2: Fraud Detection Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "scenario_2_fraud_confusion_matrix.png")
    fig.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}\n")


def run_scenario_3():
    print("=" * 60)
    print("SCENARIO 3: Group Users into Segments Based on Viewing Habits")
    print("Type: Unsupervised Learning | Clustering")
    print("Features: weekly_watch_hours, weekend_ratio, distinct_genres_watched")
    print("Target: None (discovers organic clusters)")
    print("=" * 60)

    np.random.seed(42)
    # Simulate 3 natural viewer personas
    # Persona 1: Casual weekend viewers (low hours, high weekend ratio)
    p1_hours = np.random.normal(5, 1.5, 120)
    p1_weekend = np.random.normal(0.85, 0.08, 120)

    # Persona 2: Binge watchers (high hours, moderate weekend ratio)
    p2_hours = np.random.normal(28, 4.0, 120)
    p2_weekend = np.random.normal(0.55, 0.10, 120)

    # Persona 3: Daily habitual viewers (moderate hours, low weekend ratio - watched every evening)
    p3_hours = np.random.normal(14, 2.5, 120)
    p3_weekend = np.random.normal(0.28, 0.08, 120)

    hours = np.clip(np.concatenate([p1_hours, p2_hours, p3_hours]), 1, 50)
    weekend = np.clip(np.concatenate([p1_weekend, p2_weekend, p3_weekend]), 0.05, 0.95)

    df = pd.DataFrame({
        'weekly_watch_hours': hours,
        'weekend_ratio': weekend
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Give descriptive names based on centroids
    cluster_names = {
        0: 'Cluster A',
        1: 'Cluster B',
        2: 'Cluster C'
    }
    df['cluster_label'] = df['cluster'].map(cluster_names)

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    colors = ['#0284c7', '#16a34a', '#d97706']
    for clus_id, color in zip(range(3), colors):
        subset = df[df['cluster'] == clus_id]
        ax.scatter(subset['weekly_watch_hours'], subset['weekend_ratio'],
                   label=f"Segment {clus_id + 1}", color=color, alpha=0.7, edgecolors='none')

    ax.set_title("Scenario 3: Unsupervised Viewing Habit Clusters (K=3)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Weekly Watch Hours")
    ax.set_ylabel("Weekend Ratio (Weekend vs Weekday)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "scenario_3_user_clusters.png")
    fig.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}\n")


def run_part_b():
    print("=" * 60)
    print("PART B: Custom Scenario - Predict Student Exam Pass/Fail")
    print("Type: Supervised Learning | Binary Classification")
    print("Features: attendance_pct, weekly_study_hours, assignments_completed_ratio")
    print("Target: exam_result (0 = Fail, 1 = Pass)")
    print("=" * 60)

    np.random.seed(42)
    n = 250
    attendance = np.random.uniform(40, 100, n)
    study_hours = np.random.uniform(2, 25, n)
    assignments_ratio = np.random.uniform(0.3, 1.0, n)

    # Probability of passing based on student effort
    logit = -6.0 + (0.05 * attendance) + (0.18 * study_hours) + (3.0 * assignments_ratio)
    prob = 1 / (1 + np.exp(-logit))
    pass_fail = (np.random.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        'attendance': attendance,
        'study_hours': study_hours,
        'assignments_ratio': assignments_ratio,
        'passed': pass_fail
    })

    X = df[['attendance', 'study_hours', 'assignments_ratio']]
    y = df['passed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Student Pass/Fail Prediction Accuracy: {acc:.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    scatter = ax.scatter(df['attendance'], df['study_hours'], c=df['passed'],
                         cmap='coolwarm', alpha=0.75, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Fail', 'Pass'])
    ax.set_title("Part B: Student Exam Result by Study Hours & Attendance", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Class Attendance (%)")
    ax.set_ylabel("Weekly Study Hours")
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "part_b_student_pass_fail.png")
    fig.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}\n")


if __name__ == "__main__":
    print("Starting ML Fundamentals Project Execution...")
    run_scenario_1()
    run_scenario_2()
    run_scenario_3()
    run_part_b()
    print("All models executed and charts saved successfully!")
