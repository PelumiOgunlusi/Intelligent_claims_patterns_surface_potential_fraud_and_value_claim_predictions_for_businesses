# -*- coding: utf-8 -*-
"""
Created on Sat Jul 5 2025

@author: PELUMI OGUNLUSI
This script defines a Pydantic model for fraud features, which is used to validate input data
"""
from pydantic import BaseModel


# Class which describes fraud features variables
# This class is used to validate the input data for the FastAPI endpoint
# Define the input data model
class FraudFeatures(BaseModel):
    incident_severity_Minor_Damage: str
    property_damage_Unsure: str
    incident_severity_Total_Loss: str
    policy_csl_250_500: str
    collision_type_Rear_Collision: str
    insured_sex_MALE: str
    authorities_contacted_Other: str
    incident_state_SC: str
    policy_state_IN: str
    incident_type_Single_Vehicle_Collision: str
    insured_relationship_other_relative: str
    property_damage_YES: str
    policy_state_OH: str
    bodily_injuries: str
    insured_zip: str
    total_claim_amount: str
    policy_number: str
    police_report_available_NO: str
    policy_csl_500_1000: str
    vehicle_claim: str
