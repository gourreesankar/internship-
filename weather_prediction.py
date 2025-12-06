import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def main():
    
    print("STEP 1: Loading Dataset...")
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'seattle-weather.csv')

    try:
        data = pd.read_csv(file_path)
        print(f" Data loaded successfully!")
    except FileNotFoundError:
        print(f" Error: Could not find 'seattle-weather.csv' at {file_path}")
        print("Make sure the CSV file is in the same folder as this Python file.")
        return

    
    print("STEP 2: Training the Random Forest Model (Please wait)...")
    
    
    X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
    y = data['weather']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    
    accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Model Trained! (Accuracy: {accuracy * 100:.1f}%)")

    
    print("\n" + "="*50)
    print(" WEATHER PREDICTION SYSTEM")
    print("="*50)
    print("Enter the values below. Type 'exit' to stop.\n")

    while True:
        try:
            print("\n--- Enter Weather Conditions ---")
            
            
            p_in = input("1. Precipitation (Rainfall amount): ")
            if p_in.lower().strip() == 'exit': break
            precip = float(p_in)

            
            max_in = input("2. Max Temperature: ")
            if max_in.lower().strip() == 'exit': break
            temp_max = float(max_in)

            
            min_in = input("3. Min Temperature: ")
            if min_in.lower().strip() == 'exit': break
            temp_min = float(min_in)

            
            wind_in = input("4. Wind Speed: ")
            if wind_in.lower().strip() == 'exit': break
            wind = float(wind_in)

            
            user_input = [[precip, temp_max, temp_min, wind]]
            prediction = rf_model.predict(user_input)
            result = prediction[0]

            
            print(f"\n PREDICTION: The weather is >> {result.upper()} <<")
            print("-" * 40)
            
        except ValueError:
            print("Error: Please enter valid numbers only.")
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()