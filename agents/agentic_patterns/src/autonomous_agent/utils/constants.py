TOOL_DEFINITIONS = {
    "order_lab_test": {
        "description": "Order laboratory tests",
        "parameters": ["test_name", "priority", "instructions"]
    },
    "refer_specialist": {
        "description": "Make specialist referrals",
        "parameters": ["specialty", "priority", "reason"]
    },
    "prescribe_medication": {
        "description": "Prescribe medications",
        "parameters": ["medication", "dosage", "frequency", "duration"]
    },
    "schedule_followup": {
        "description": "Schedule follow-up appointments",
        "parameters": ["department", "timeframe", "priority"]
    }
}

PATIENT_STATES = [
    "STABLE",
    "REQUIRES_MONITORING",
    "URGENT",
    "CRITICAL"
]

DEFAULT_TEMPERATURE = 0.3

SAMPLE_CASE = """73-year-old female presents with increasing shortness of breath over past 2 weeks,
chest pain on exertion, and swelling in both legs. History of hypertension, diabetes, and atrial fibrillation.
Current medications include metformin and lisinopril. BP 158/92, HR 92 irregular, SpO2 93%, Temperature 37.2°C.
Recent CBC shows mild anemia. Patient reports difficulty sleeping due to orthopnea.
Has been self-adjusting medications due to symptoms"""