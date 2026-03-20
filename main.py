import sys
from src.preprocess import load_and_preprocess_data
from src.train_model import train_and_evaluate
from src.predict import predict_cgpa
import pandas as pd

def main():
    print("Education CGPA Prediction")
    print("Select an option:")
    print("1. Train Model")
    print("2. Predict CGPA for new data")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("--- Training Model ---")
        train_and_evaluate()
    elif choice == "2":
        print("--- Predicting CGPA ---")
        sample_input = input("Enter path to new student data file (CSV or XLSX): ").strip()
        # Support both CSV and Excel formats for input
        if sample_input.lower().endswith(".xlsx"):
            new_data = pd.read_excel(sample_input)
        else:
            new_data = pd.read_csv(sample_input)
        # Drop Student_ID if present
        if "Student_ID" in new_data.columns:
            new_data = new_data.drop("Student_ID", axis=1)
        # Fill missing values and encode categorical columns
        for col in new_data.columns:
            if new_data[col].dtype == "object":
                new_data[col] = new_data[col].fillna(new_data[col].mode()[0])
            else:
                new_data[col] = new_data[col].fillna(new_data[col].mean())
        new_data = pd.get_dummies(new_data, columns=["Gender", "Course", "Living_Type", "Club_Participation", "Counseling_Access"], drop_first=True)
        # Ensure columns match model expectations
        training_cols = list(load_and_preprocess_data("data/Education_Dataset.xlsx")[5])
        for col in training_cols:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[training_cols]
        # Select only the first row for prediction:
        single_student = new_data.iloc[[0]]
        result = predict_cgpa(single_student)
        print("Predicted CGPA for first student:", result[0])
    else:
        print("Invalid selection! Please enter 1 or 2.")

if __name__ == "__main__":
    main()
