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
    incident_severity_Minor_Damage: int
    property_damage_Unsure: int
    incident_severity_Total_Loss: int
    policy_csl_250_500: int
    collision_type_Rear_Collision: int
    insured_sex_MALE: int
    authorities_contacted_Other: int
    incident_state_SC: int
    policy_state_IN: int
    incident_type_Single_Vehicle_Collision: int
    insured_relationship_other_relative: int
    property_damage_YES: int
    policy_state_OH: int
    bodily_injuries: int
    insured_zip: int
    total_claim_amount: int
    policy_number: int
    police_report_available_NO: int
    policy_csl_500_1000: int
    vehicle_claim: int
