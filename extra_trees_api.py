# -*- coding: utf-8 -*-
"""
Created on Sat July 5 2025

@author: PELUMI OGUNLUSI
This script sets up a FastAPI application to serve predictions from a pre-trained Extra Trees model for fraud detection.
It includes an endpoint to receive input data, make predictions, and return the results.
"""

# Library imports
from fastapi import FastAPI
from fraud_cases import FraudFeatures
import joblib
import uvicorn
import numpy as np
import pandas as pd

# Create the app object
app = FastAPI()

# Load the pre-trained Extra Trees model
# Ensure the model is in the same directory or provide the correct path
model = joblib.load("extra_trees_model.pkl")


# Home route for health check, opens automatically when the server starts http://127.0.0.1:8000
@app.get("/")
def home():
    return {"Welcome to this fraud detection model": "OK"}


# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted fraudulent case probability with confidence
@app.post("/predict")
def predict_fraud(data: FraudFeatures):
    data = data.dict()
    incident_severity_Minor_Damage = data["incident_severity_Minor_Damage"]
    property_damage_Unsure = data["property_damage_Unsure"]
    incident_severity_Total_Loss = data["incident_severity_Total_Loss"]
    policy_csl_250_500 = data["policy_csl_250_500"]
    collision_type_Rear_Collision = data["collision_type_Rear_Collision"]
    insured_sex_MALE = data["insureinsured_sex_MALE"]
    authorities_contacted_Other = data["authorities_contacted_Other"]
    incident_state_SC = data["incident_state_SC"]
    policy_state_IN = data["policy_state_IN"]
    incident_type_Single_Vehicle_Collision = data["incident_type_Single_Vehicle_Collision"]   
    insured_relationship_other_relative = data["insured_relationship_other_relative"]
    property_damage_YES = data["property_damage_YES"]
    policy_state_OH = data["policy_state_OH"]
    bodily_injuries = data["bodily_injuries"]
    insured_zip = data["insured_zip"]
    total_claim_amount = data["total_claim_amount"]
    policy_number = data["policy_number"]
    police_report_available_NO = data["police_report_available_NO"]
    policy_csl_500_1000 = data["policy_csl_500_1000"]
    vehicle_claim = data["vehicle_claim"]
    prediction = model.predict(
        [
            [
                incident_severity_Minor_Damage,
                property_damage_Unsure,
                incident_severity_Total_Loss,
                policy_csl_250_500,
                collision_type_Rear_Collision,
                insured_sex_MALE,
                authorities_contacted_Other,
                incident_state_SC,
                policy_state_IN,
                incident_type_Single_Vehicle_Collision,
                insured_relationship_other_relative,
                property_damage_YES,
                policy_state_OH,
                bodily_injuries,
                insured_zip,
                total_claim_amount,
                policy_number,
                police_report_available_NO,
                policy_csl_500_1000,
                vehicle_claim
            ]
        ]
    )
    if prediction[0] > 0.5:
        prediction = "Fraudulent Case"
    else:
        prediction = "Non-Fraudulent Case"
    return {"prediction": prediction}


#  Run the API with uvicorn
#  Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn extra_trees_api:app --reload (To run the API server from the command line/terminal)
