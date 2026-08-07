import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# MLflow Configuration
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Prediction")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/heart.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_accuracy = 0
best_model = None

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# -----------------------------
# Train and Evaluate
# -----------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy : {accuracy:.4f}")

        # Log parameters
        mlflow.log_param("Model", name)
        mlflow.log_param("Test Size", 0.2)
        mlflow.log_param("Random State", 42)

        # Log metrics
        mlflow.log_metric("Accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:5]
        )

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, "models/model.pkl")

print("\n-------------------------------")
print("Best Accuracy :", best_accuracy)
print("Best model saved successfully!")
print("-------------------------------")