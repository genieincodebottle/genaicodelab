MEDICAL_ROUTES = {
    "urgent_care": """You are an urgent care triage specialist. Follow these guidelines:
    1. Start with "Urgent Care Assessment:"
    2. Evaluate symptoms for severity
    3. Determine appropriate level of care
    4. Provide immediate action steps
    5. Include red flag warnings if applicable""",

    "chronic_care": """You are a chronic care coordinator. Follow these guidelines:
    1. Start with "Chronic Care Management:"
    2. Review ongoing conditions
    3. Assess medication compliance
    4. Evaluate lifestyle factors
    5. Provide long-term management strategies""",

    "preventive_care": """You are a preventive care specialist. Follow these guidelines:
    1. Start with "Preventive Care Recommendations:"
    2. Review risk factors
    3. Suggest screening tests
    4. Provide lifestyle recommendations
    5. Set prevention goals""",

    "specialist_referral": """You are a medical referral coordinator. Follow these guidelines:
    1. Start with "Specialist Referral Assessment:"
    2. Identify specialty need
    3. Review case complexity
    4. Determine urgency
    5. Provide referral process steps"""
}

HEALTHCARE_STAKEHOLDERS = [
    """Patient Care Team:
    - Primary care physician
    - Specialist consultants
    - Nursing staff
    - Allied health professionals""",

    """Patient and Family:
    - Patient's immediate needs
    - Family caregiver requirements
    - Home care considerations
    - Financial implications""",

    """Healthcare Facility:
    - Resource allocation
    - Staff scheduling
    - Equipment needs
    - Regulatory compliance""",

    """Support Services:
    - Laboratory
    - Radiology
    - Pharmacy
    - Physical therapy"""
]

SAMPLE_MEDICAL_REPORT = """
Patient Visit Summary:
73-year-old female presents with increasing shortness of breath over past 2 weeks.
BP 138/82, HR 88, RR 20, T 37.2°C, SpO2 94% on room air.
Reports fatigue and mild chest pain on exertion.
History of hypertension and type 2 diabetes.
Current medications include Metformin 1000mg BID and Lisinopril 10mg daily.
Physical exam reveals mild bilateral ankle edema and decreased breath sounds at bases.
ECG shows normal sinus rhythm with nonspecific ST-T wave changes.
Basic labs ordered including CBC, CMP, and BNP.
Patient scheduled for follow-up echocardiogram.
"""

SAMPLE_MEDICAL_QUERIES = [
    """Patient Case:
    58-year-old male experiencing severe chest pain radiating to left arm for past hour.
    History of hypertension. No prior cardiac events.
    Sweating and nauseous. BP 160/95.
    Requesting guidance on immediate steps.""",

    """Patient Case:
    42-year-old female with type 2 diabetes for 5 years.
    Recent A1C 8.2%, up from 7.1% three months ago.
    Reports difficulty following diet plan and checking glucose regularly.
    Seeking management strategy adjustment.""",

    """Patient Case:
    35-year-old female, no significant medical history.
    Due for annual check-up. Last mammogram 2 years ago.
    Family history of breast cancer (mother, age 45).
    Requesting preventive care planning."""
]