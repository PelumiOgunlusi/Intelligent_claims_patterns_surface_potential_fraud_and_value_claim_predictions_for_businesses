from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np
import pandas as pd
import sklearn

app = FastAPI()
# Define the input data model
class FraudFeatures(BaseModel):
    collision_type_Rear Collision: bool
    incident_severity_Total Loss: bool
    incident_severity_Minor Damage: bool
    incident_state_SC: float
    property_damage_Unsure: str
    insured_sex_MALE: bool
    policy_csl_250/500: bool
    incident_type_Single Vehicle Collision: bool
    policy_state_OH: bool
    property_damage_YES: bool
    authorities_contacted_Other : bool
    policy_state_IN : bool
    insured_relationship_other-relative: bool
    insured_zip: int
    total_claim_amount: int
    vehicle_claim: int
    policy_csl_500/1000: bool
    property_claim: int


# Load the pre-trained Extra Trees model

# Ensure the model is in the same directory or provide the correct path

model = joblib.load("extra_trees_model.pkl")

@app.get("/")
def home():
return {"health_check": "OK"}


@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)
    return {"fraud_reported": int(pred[0])}


if __name__ == "__main__":
    # Start the FastAPI server  
    # Use uvicorn to run the app
    
    uvicorn.run("extra_trees_api:app", host="0.0.0.0", port=8000, reload=True)
