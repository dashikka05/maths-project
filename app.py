
from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("diabetes_risk_assessment_dataset(3).csv")

# Input features and target
features = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level",
    "blood_glucose_level"
]

X = df[features]
y = df["diabetes"]

# Numerical and categorical columns
numeric = [
    "age", "bmi", "HbA1c_level", "blood_glucose_level",
    "hypertension", "heart_disease"
]
categorical = ["gender", "smoking_history"]

# Prepare data mathematically
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train and test the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None

    if request.method == "POST":
        person = pd.DataFrame([{
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "smoking_history": request.form["smoking_history"],
            "bmi": float(request.form["bmi"]),
            "HbA1c_level": float(request.form["HbA1c_level"]),
            "blood_glucose_level": float(
                request.form["blood_glucose_level"]
            )
        }])[features]

        prediction = model.predict(person)[0]
        probability = round(
            model.predict_proba(person)[0][1] * 100, 2
        )

        result = (
            "Higher predicted diabetes risk"
            if prediction == 1
            else "Lower predicted diabetes risk"
        )

    return render_template(
        "index.html", result=result, probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)

