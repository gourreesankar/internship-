import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
def load_and_prep_data(filename):
    try:
        df = pd.read_csv(filename)
        print(f" Loaded file: {filename} with {len(df)} rows.")
    except FileNotFoundError:
        print(f" Error: Could not find file '{filename}'. Make sure it is in the same folder.")
        return None
    except Exception as e:
        print(f" Error: An unexpected error occurred: {e}")
        return None

    feature_cols = [
        'Price/Earnings', 
        'Dividend Yield', 
        'Earnings/Share', 
        '52 Week Low', 
        '52 Week High']
    target_col = 'Price'

    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        print(f" Error: Missing columns in CSV: {missing}")
        return None
    data = df[feature_cols + [target_col]].copy()

    data = data.dropna()
    print(f" Data after cleaning: {len(data)} rows ready for training.")
    return data, feature_cols, target_col
def main():
    
    result = load_and_prep_data('project1.csv')
    if result is None:
        return
    df, features, target = result
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n Training Random Forest Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("\n Model Results:")
    print(f"   Accuracy (R2 Score): {accuracy:.2f} (1.00 is perfect)")
    print(f"   Average Error: ${mae:.2f}")

    print("\n What drove the price most?")
    importances = model.feature_importances_
    for name, importance in zip(features, importances):
        print(f"   {name}: {importance:.4f}")
    
    plt.figure(figsize=(10, 6))

    limit = 50
    plt.plot(y_test.values[:limit], label='Actual Price', marker='o')
    plt.plot(predictions[:limit], label='Predicted Price (Fair Value)', linestyle='--', marker='x')

    plt.title('Actual Stock Price vs. Model Predicted Price')
    plt.xlabel('Companies (Sample Index)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()






