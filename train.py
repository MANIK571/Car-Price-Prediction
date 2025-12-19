# ==========================================
# Comparison of ML Models - Save All Models
# Car Price Prediction
# ==========================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pickle

# ------------------------------------------
# 1. Create folder to save models
# ------------------------------------------
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------
# 2. Load Dataset
# ------------------------------------------
df = pd.read_csv("car data.csv")
df.drop("Car_Name", axis=1, inplace=True)

# ------------------------------------------
# 3. Split Features & Target
# ------------------------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ------------------------------------------
# 4. Preprocessing
# ------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("fuel", OneHotEncoder(drop="first"), ["Fuel_Type"]),
        ("seller", OneHotEncoder(drop="first"), ["Seller_Type"]),
        ("trans", OneHotEncoder(drop="first"), ["Transmission"])
    ],
    remainder="passthrough"
)

# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# 6. Define Models
# ------------------------------------------
models = {
    "Linear_Regression": LinearRegression(),
    "Ridge_Regression": Ridge(alpha=1.0),
    "Lasso_Regression": Lasso(alpha=0.01),
    "Decision_Tree": DecisionTreeRegressor(random_state=42),
    "Random_Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=42)
}

# ------------------------------------------
# 7. Train, Evaluate & Save All Models
# ------------------------------------------
results = []

best_model = None
best_accuracy = -1

for name, regressor in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", regressor)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Save each model
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    pickle.dump(pipeline, open(model_path, "wb"))

    results.append([name, accuracy, r2, mae, rmse])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

# ------------------------------------------
# 8. Save Best Model Separately
# ------------------------------------------
pickle.dump(best_model, open("best_car_price_model.pkl", "wb"))

# ------------------------------------------
# 9. Display Comparison Results
# ------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy (%)", "R2 Score", "MAE", "RMSE"]
)

print("\n MODEL COMPARISON RESULTS \n")
print(results_df.sort_values(by="Accuracy (%)", ascending=False))

print("\n All models saved in 'saved_models/' folder")
print(f" Best Model: {best_model_name} ({best_accuracy:.2f}%)")
print(" Best model also saved as 'best_car_price_model.pkl'")
