import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(input_path):
    df = pd.read_excel(input_path)
    # Drop Student_ID (not useful for prediction)
    if "Student_ID" in df.columns:
        df = df.drop("Student_ID", axis=1)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categoricals
    df = pd.get_dummies(df, columns=["Gender", "Course", "Living_Type", "Club_Participation", "Counseling_Access"], drop_first=True)
    X = df.drop("CGPA", axis=1)
    y = df["CGPA"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, X.columns

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data("../data/Education_Dataset.csv")
    print("Preprocessing successful. Feature columns:", list(feature_names))
