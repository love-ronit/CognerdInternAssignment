from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import sqlite3
import datetime
import pickle
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inference_api")

app = FastAPI(title="Inference API")

# Global variables to hold model and features
model = None
feature_info = None

# Database setup
DB_FILE = 'predictions.db'

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_features TEXT,
                prediction REAL,
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Pydantic model for input validation
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.on_event("startup")
async def startup_event():
    global model, feature_info
    logger.info("Service starting up...")
    
    # Load model
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
        
    # Load feature info
    try:
        with open('features.pkl', 'rb') as f:
            feature_info = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load feature info: {e}. Validation might be limited.")
        
    init_db()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(features: HouseFeatures, request: Request):
    logger.info(f"Received predict request to {request.url.path}")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert Pydantic model to dictionary
        input_data = features.dict()
        
        # Feature Engineering
        bedrooms_per_room = input_data['AveBedrms'] / input_data['AveRooms'] if input_data['AveRooms'] > 0 else 0
        is_new_house = 1 if input_data['HouseAge'] < 10 else 0
        
        # Prepare the feature array in the correct order
        # Expected order from training: 
        # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, BedroomsPerRoom, IsNewHouse
        if feature_info:
            # Reconstruct the feature DataFrame or list
            import pandas as pd
            # Create a single-row DataFrame with raw features
            df = pd.DataFrame([input_data])
            df['BedroomsPerRoom'] = bedrooms_per_room
            df['IsNewHouse'] = is_new_house
            # Ensure columns are in the same order as trained
            X_input = df[feature_info["engineered_features"]]
        else:
            # Fallback if features.pkl is missing
            import pandas as pd
            input_data['BedroomsPerRoom'] = bedrooms_per_room
            input_data['IsNewHouse'] = is_new_house
            X_input = pd.DataFrame([input_data])

        # Make prediction
        prediction_value = float(model.predict(X_input)[0])
        
        # Confidence is not applicable for Linear Regression
        confidence = None
        
        # Save to database
        timestamp = datetime.datetime.now().isoformat()
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (timestamp, input_features, prediction, confidence) VALUES (?, ?, ?, ?)",
                (timestamp, str(input_data), prediction_value, confidence)
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.error(f"Database write failed: {db_err}")
            # Do not break response if DB write fails
            
        return {
            "prediction": prediction_value
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
