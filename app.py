import streamlit as st
import requests
import os

# Ensure localhost requests don't get routed through system proxies (like Fivetran webhooks)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

st.title("California Housing Price Predictor")
st.write("Enter the features of the house to predict its median value (in $100,000s).")

# API Endpoint
API_URL = "http://localhost:8000/predict"

# Input form
with st.form("prediction_form"):
    st.subheader("House Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        MedInc = st.number_input("Median Income (in $10k)", min_value=0.0, value=3.5, step=0.1)
        HouseAge = st.number_input("House Age (years)", min_value=0.0, value=20.0, step=1.0)
        AveRooms = st.number_input("Average Rooms", min_value=1.0, value=5.0, step=0.1)
        AveBedrms = st.number_input("Average Bedrooms", min_value=1.0, value=1.0, step=0.1)
        
    with col2:
        Population = st.number_input("Population", min_value=1.0, value=300.0, step=10.0)
        AveOccup = st.number_input("Average Occupancy", min_value=1.0, value=3.0, step=0.1)
        Latitude = st.number_input("Latitude", value=35.0, step=0.1)
        Longitude = st.number_input("Longitude", value=-120.0, step=0.1)
        
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Prepare the payload
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }
    
    with st.spinner("Connecting to Inference API..."):
        try:
            # Send POST request
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction")
                st.success(f"Predicted Median House Value: **${prediction * 100000:,.2f}**")
            elif response.status_code == 400:
                st.error(f"Invalid input: {response.text}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to the Inference API. Is it running on http://localhost:8000?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
