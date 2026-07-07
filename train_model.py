import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def engineer_features(df):
    """
    Crafts two new features as requested:
    1. BedroomsPerRoom: Ratio of bedrooms to total rooms.
    2. IsNewHouse: Binary flag indicating if the house age is less than 10 years.
    """
    df = df.copy()
    # Feature 1: Ratio feature
    df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
    
    # Feature 2: Binning/Flag feature
    df['IsNewHouse'] = (df['HouseAge'] < 10).astype(int)
    
    return df

def main():
    print("Loading California Housing dataset...")
    california = fetch_california_housing()
    
    # Create DataFrame
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    
    print("Original features:", X.columns.tolist())
    
    # Feature engineering
    print("Engineering new features...")
    X_engineered = engineer_features(X)
    print("Features after engineering:", X_engineered.columns.tolist())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)
    
    # Create a simple pipeline: scaling + linear regression
    # (Since this is a simple ML problem and we need something lightweight)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation - MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # Save the model
    model_path = 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # We also want to expose the expected feature names for validation in the API
    feature_info = {
        "raw_features": california.feature_names,
        "engineered_features": X_engineered.columns.tolist()
    }
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
if __name__ == "__main__":
    main()
