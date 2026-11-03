
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. FLASK SETUP
# ==========================================
app = Flask(__name__, static_folder="graphs")
os.makedirs("graphs", exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================
data = pd.read_csv("diabetes_risk_assessment_dataset.csv")
data.columns = data.columns.str.strip().str.lower()

features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "hba1c_level",
    "blood_glucose_level"
]
target = "diabetes"

data = data[features + [target]].apply(
    pd.to_numeric, errors="coerce"
).dropna()

data = data[data[target].isin([0, 1])]

X = data[features]
y = data[target].astype(int)

if y.nunique() != 2:
    raise ValueError(
        "The diabetes column must contain both 0 and 1."
    )

# ==========================================
# 3. TRAIN LOGISTIC REGRESSION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(
    y_test, y_pred, zero_division=0
) * 100
recall = recall_score(
    y_test, y_pred, zero_division=0
) * 100
f1 = f1_score(
    y_test, y_pred, zero_division=0
) * 100

# Specificity: correctly predicted negative cases
tn = int(((y_test == 0) & (y_pred == 0)).sum())
fp = int(((y_test == 0) & (y_pred == 1)).sum())

specificity = (
    tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
)

# ==========================================
# 4. STATISTICAL SUMMARY
# ==========================================
stats = data[features].agg(
    ["mean", "median", "std", "var", "min", "max"]
).round(2)

counts = y.value_counts().reindex(
    [0, 1], fill_value=0
)

# ==========================================
# 5. GRAPH: CLASS DISTRIBUTION
# ==========================================
plt.figure(figsize=(6, 4))

counts.rename(
    index={0: "No diabetes", 1: "Diabetes"}
).plot(kind="bar")

plt.title("Diabetes Class Distribution")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/classes.png")
plt.close()

# ==========================================
# 6. GRAPH: CORRELATION HEATMAP
# ==========================================
plt.figure(figsize=(8, 6))

sns.heatmap(
    data[features + [target]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation.png")
plt.close()

# ==========================================
# 7. GRAPH: AVERAGE GLUCOSE
# ==========================================
glucose = data.groupby(target)[
    "blood_glucose_level"
].mean()

glucose.index = [
    "No diabetes" if i == 0 else "Diabetes"
    for i in glucose.index
]

plt.figure(figsize=(6, 4))
glucose.plot(kind="bar")
plt.title("Average Blood Glucose Level")
plt.ylabel("Average Glucose")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/glucose.png")
plt.close()

# ==========================================
# 8. GRAPH 4: LOGISTIC SIGMOID FUNCTION
# ==========================================
z_values = np.linspace(-8, 8, 300)
probabilities = 1 / (1 + np.exp(-z_values))

plt.figure(figsize=(7, 4))
plt.plot(z_values, probabilities, label="Sigmoid function")
plt.axhline(
    0.5, color="red", linestyle="--",
    label="50% decision threshold"
)
plt.axvline(0, color="gray", linestyle=":")
plt.xlabel("Linear score (z)")
plt.ylabel("Probability")
plt.title("Logistic Regression Sigmoid Function")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/sigmoid.png")
plt.close()

# ==========================================
# 9. GRAPH 5: FEATURE ANALYSIS
# ==========================================
logistic = model.named_steps["logisticregression"]

coefficients = pd.Series(
    logistic.coef_[0],
    index=features
).sort_values()

plt.figure(figsize=(8, 5))
coefficients.plot(kind="barh")
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Standardized Model Coefficient")
plt.title("Feature Analysis")
plt.tight_layout()
plt.savefig("graphs/feature_analysis.png")
plt.close()

feature_rows = []

for name, value in coefficients.sort_values(
    ascending=False
).items():
    if value > 0:
        direction = "Positive model association"
    elif value < 0:
        direction = "Negative model association"
    else:
        direction = "Zero coefficient"

    feature_rows.append({
        "name": name.replace("_", " ").title(),
        "coefficient": round(float(value), 4),
        "direction": direction
    })

# ==========================================
# 10. HOME PAGE AND INDIVIDUAL PREDICTION
# ==========================================
@app.route("/", methods=["GET", "POST"])
def home():
    report = None
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form[name])
                for name in features
            ]

            if not all(np.isfinite(v) for v in values):
                raise ValueError

            if any(v < 0 for v in values):
                error = "Values cannot be negative."

            elif values[1] not in [0, 1] or values[2] not in [0, 1]:
                error = (
                    "Hypertension and heart disease "
                    "must be 0 or 1."
                )

            elif (
                values[0] > 120
                or values[3] > 100
                or values[4] > 20
                or values[5] > 600
            ):
                error = "Please check the entered values."

            else:
                person = pd.DataFrame(
                    [values], columns=features
                )

                # Probability and predicted category
                probability = float(
                    model.predict_proba(person)[0][1]
                )
                result = int(model.predict(person)[0])

                # Standardize the input for the equation
                scaler = model.named_steps["standardscaler"]
                scaled_values = scaler.transform(person)[0]

                # z = b0 + b1*x1 + ... + b6*x6
                z = float(
                    np.dot(
                        scaled_values,
                        logistic.coef_[0]
                    ) + logistic.intercept_[0]
                )

                # Sigmoid probability calculation
                calculated_probability = float(
                    1 / (1 + np.exp(-np.clip(z, -500, 500)))
                )

                # Individual prediction probability graph
                plt.figure(figsize=(7, 4))
                plt.bar(
                    ["Estimated probability"],
                    [probability * 100]
                )
                plt.axhline(
                    50,
                    color="red",
                    linestyle="--",
                    label="50% decision threshold"
                )
                plt.ylim(0, 100)
                plt.ylabel("Probability (%)")
                plt.title("Individual Prediction Probability")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    "graphs/individual_prediction.png"
                )
                plt.close()

                input_report = [
                    {
                        "name": name.replace("_", " ").title(),
                        "value": round(value, 2)
                    }
                    for name, value in zip(features, values)
                ]

                report = {
                    "category": (
                        "Diabetes-positive category"
                        if result == 1
                        else "Diabetes-negative category"
                    ),
                    "probability": round(probability * 100, 2),
                    "z_score": round(z, 4),
                    "calculated_probability": round(
                        calculated_probability * 100, 2
                    ),
                    "inputs": input_report
                }

        except (ValueError, TypeError, OverflowError):
            error = "Please enter valid numerical values."

    return render_template(
        "index.html",
        stats=stats.to_dict(),
        accuracy=round(accuracy, 2),
        precision=round(precision, 2),
        recall=round(recall, 2),
        specificity=round(specificity, 2),
        f1=round(f1, 2),
        records=len(data),
        diabetes_count=int(counts[1]),
        no_diabetes_count=int(counts[0]),
        feature_rows=feature_rows,
        report=report,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
