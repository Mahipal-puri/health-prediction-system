"""
Smart Healthcare Risk Prediction System
Model Training Script

Trains an MLP Neural Network and Linear Regression model
on the Medical Cost Personal Dataset to predict patient risk levels.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_and_preprocess(csv_path):
    """Load dataset and perform all preprocessing steps."""
    df = pd.read_csv(csv_path)

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Create risk target variable
    df["risk"] = (df["charges"] > 15000).astype(int)

    # Encode categorical variables
    df["sex_encoded"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker_encoded"] = df["smoker"].map({"yes": 1, "no": 0})

    # One-hot encode region
    region_dummies = pd.get_dummies(df["region"], prefix="region", dtype=int)
    df = pd.concat([df, region_dummies], axis=1)

    return df


def train_models(df, model_dir):
    """Train MLP classifier and Linear Regression model."""
    feature_cols = [
        "age", "bmi", "children", "sex_encoded", "smoker_encoded",
        "region_northeast", "region_northwest", "region_southeast", "region_southwest",
    ]

    X = df[feature_cols].values
    y_class = df["risk"].values
    y_reg = df["charges"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train_c, y_test_c = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        df[feature_cols], y_reg, test_size=0.2, random_state=42
    )

    # ---- MLP Neural Network ----
    print("=" * 60)
    print("TRAINING MLP NEURAL NETWORK")
    print("=" * 60)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        alpha=0.0001,
    )
    mlp.fit(X_train, y_train_c)

    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test_c, y_pred)

    print(f"Architecture: Input({X_train.shape[1]}) -> Hidden(64) -> Hidden(32) -> Hidden(16) -> Output(1)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Iterations: {mlp.n_iter_}")
    print(f"Final Loss: {mlp.loss_:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_c, y_pred, target_names=["Low Risk", "High Risk"]))

    # ---- Linear Regression ----
    print("=" * 60)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 60)

    lr = LinearRegression()
    lr.fit(X_train_r, y_train_r)
    y_pred_r = lr.predict(X_test_r)

    print(f"R² Score: {r2_score(y_test_r, y_pred_r):.4f}")
    print(f"MAE: ₹{mean_absolute_error(y_test_r, y_pred_r) * 83:,.2f}")
    print(f"RMSE: ₹{np.sqrt(mean_squared_error(y_test_r, y_pred_r)) * 83:,.2f}")

    # ---- Save models ----
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(mlp, os.path.join(model_dir, "mlp_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(lr, os.path.join(model_dir, "lr_model.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))

    print(f"\nAll models saved to: {model_dir}")
    return mlp, lr, scaler, feature_cols


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "dataset", "insurance.csv")
    model_dir = script_dir

    print("Loading dataset...")
    df = load_and_preprocess(csv_path)
    print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
    print(f"Risk distribution: Low={int((df['risk'] == 0).sum())}, High={int(df['risk'].sum())}\n")

    train_models(df, model_dir)
    print("\nTraining complete!")
