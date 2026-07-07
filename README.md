# Inference API + Streamlit UI

This is a miniature production ML service containing a REST inference API and a Streamlit web app. The service is built with FastAPI, Streamlit, and `scikit-learn`.

## 1. Environment Setup

To run the project, set up a Python virtual environment and install the dependencies:

```bash
python -m venv venv
```
```bash
# On Windows:
.\venv\Scripts\Activate
```
```bash
# On Linux/macOS:
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

Generate the model file before starting the API:
```bash
python train_model.py
```

Execution Policy
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. How to start the API

The inference API is built using FastAPI. It exposes a `/health` endpoint and a `/predict` endpoint, and saves all predictions to a local SQLite database (`predictions.db`).

Start the API with Uvicorn:
```bash
uvicorn api:app --reload
# or
python -m uvicorn api:app --reload
```
The API will run at `http://localhost:8000`.

## 3. How to start the Streamlit app

The Streamlit app provides a user-friendly form to query the inference API.

Start the app in a new terminal (with the virtual environment activated):
```bash
streamlit run app.py
```
This will open the app in your default browser at `http://localhost:8501`.

## 4. Test the API directly (curl)

You can test the `/predict` endpoint directly using `curl`:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"MedInc\": 3.5, \"HouseAge\": 20.0, \"AveRooms\": 5.0, \"AveBedrms\": 1.0, \"Population\": 300.0, \"AveOccup\": 3.0, \"Latitude\": 35.0, \"Longitude\": -120.0}"
```

## 5. Design Decisions & Trade-offs

### Problem & Dataset
We chose the **California Housing dataset**, modeling it as a regression problem to predict the median house value based on various housing features. This is a simple, robust dataset perfectly suited for demonstrating an end-to-end inference service without overly complex modeling that would distract from the engineering side of the assignment. 

### Feature Engineering & Modeling
To satisfy the feature engineering requirement, we crafted two new features:
- `BedroomsPerRoom` (Derived/Ratio): `AveBedrms / AveRooms`. This shows the concentration of bedrooms within a house, which can influence pricing.
- `IsNewHouse` (Binning/Flag): `1` if `HouseAge < 10` else `0`. Age bands significantly affect perceived property value.

For the model, we chose a simple `scikit-learn` pipeline with a `StandardScaler` and `LinearRegression`. This ensures extremely fast training and inference times while remaining highly robust to missing values or outlier predictions, emphasizing system design and persistence (via SQLite) over chasing model accuracy metrics.

### Future Improvements
With more time, I would containerize the service using Docker and `docker-compose` so that the frontend, backend, and a dedicated database (e.g., PostgreSQL instead of SQLite) run in unison. I would also add more comprehensive unit tests for the API using `pytest` and implement better tracking for data drift on the input features over time.
