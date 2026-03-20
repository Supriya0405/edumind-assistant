from src.preprocess import load_and_preprocess_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
os.makedirs("model", exist_ok=True)



def train_and_evaluate():
    

    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data("data/Education_Dataset.xlsx")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.3f}")
    print(f"Test R2 Score: {r2:.3f}")
    # Save model and scaler for later use
    joblib.dump(model, "model/random_forest_regressor.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_and_evaluate()
