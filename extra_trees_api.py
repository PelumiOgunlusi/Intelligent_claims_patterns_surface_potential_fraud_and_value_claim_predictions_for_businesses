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
    collision_type_Rear_Collision: bool
    incident_severity_Total_Loss: bool
    incident_severity_Minor_Damage: bool
    incident_state_SC: bool
    property_damage_Unsure: bool
    insured_sex_MALE: bool
    policy_csl_250_500: bool
    incident_type_Single_Vehicle_Collision: bool
    policy_state_OH: bool
    property_damage_YES: bool
    authorities_contacted_Other: bool
    police_report_available_NO: bool
    collision_type_Side_Collision: bool
    insured_zip: int
    total_claim_amount: int
    vehicle_claim: int
    witnesses: int
    property_claim: int


# Load the pre-trained Extra Trees model

# Ensure the model is in the same directory or provide the correct path

model = joblib.load("extra_trees_model.pkl")


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict")
async def predict(data: FraudFeatures):
    try:
        df = pd.DataFrame([data.model_dump()])
        pred = model.predict(df)
        return {"fraud_reported": int(pred[0])}
    except Exception as e:
        # Handle any exceptions that occur during prediction
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    # Start the FastAPI server
    # Use uvicorn to run the app
    uvicorn.run("extra_trees_api:app", host="0.0.0.0", port=8000, reload=True)
