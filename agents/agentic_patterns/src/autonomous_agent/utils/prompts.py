# medical_prompts.py

ANALYSIS_PROMPT = """Analyze this medical case as a clinical expert:

Patient Information:
{case}

Provide a structured analysis considering:
1. Immediate clinical concerns
2. Risk factors
3. Required monitoring
4. Preliminary diagnosis considerations

Format your response exactly as follows with NO line breaks in the XML content:
<analysis>{{"patient_state":"STABLE","immediate_concerns":[],"risk_factors":[],"monitoring_needs":[],"preliminary_diagnosis":[],"reasoning":"your reasoning here"}}</analysis>

Note: patient_state must be one of: STABLE, REQUIRES_MONITORING, URGENT, CRITICAL
"""

TASK_PLANNING_PROMPT = """Create a medical task plan based on this analysis:

Patient State: {patient_state}
Analysis: {analysis}
Available Tools: {tools}

You must create a comprehensive task list that includes appropriate tasks using the available tools.

Required format:
You must respond with a valid JSON array of tasks wrapped in tasks XML tags. Each task must have these fields:
- id: string (e.g., "task_1")
- description: string describing the task
- priority: number (1 = highest priority)
- required_tools: array of tool names from the available tools
- dependencies: array of task IDs that must be completed before this task

Example response structure:
<tasks>[
    {{
        "id": "task_1",
        "description": "Initial lab tests",
        "priority": 1,
        "required_tools": ["order_lab_test"],
        "dependencies": []
    }},
    {{
        "id": "task_2",
        "description": "Specialist consultation",
        "priority": 2,
        "required_tools": ["refer_specialist"],
        "dependencies": ["task_1"]
    }}
]</tasks>

Important:
- All tasks must have unique IDs
- Only use tools from the provided Available Tools list
- Do not include line breaks in the JSON
- Dependencies must only reference existing task IDs
- Priority should be 1 for most urgent tasks
"""

LAB_TEST_PROMPT = """Order appropriate laboratory tests based on the current task:

Task: {task_description}
Patient State: {patient_state}
Current Analysis: {analysis}

Format your response exactly as follows with NO line breaks in the XML content:
<orders>{{"test_orders":[{{"test_name":"test name","priority":"routine","instructions":"instructions"}}],"rationale":"your rationale"}}</orders>
"""

SPECIALIST_REFERRAL_PROMPT = """Generate specialist referrals based on the current task:

Task: {task_description}
Patient State: {patient_state}
Clinical Context: {context}

Format your response exactly as follows with NO line breaks in the XML content:
<referrals>{{"specialist_referrals":[{{"specialty":"specialty name","priority":"routine","reason":"reason","notes":"notes"}}],"rationale":"your rationale"}}</referrals>
"""

PRESCRIPTION_PROMPT = """Prescribe appropriate medications based on the current task:

Task: {task_description}
Patient State: {patient_state}
Current Medications: {medications}
Clinical Context: {context}

Format your response exactly as follows with NO line breaks in the XML content:
<prescriptions>{{"medication_orders":[{{"medication":"med name","dosage":"dosage","frequency":"frequency","duration":"duration","special_instructions":"instructions"}}],"rationale":"your rationale"}}</prescriptions>
"""

CARE_PLAN_PROMPT = """Create a comprehensive care plan based on completed tasks:

Analysis: {analysis}
Completed Tasks: {completed_tasks}
Task Results: {task_results}

Format your response exactly as follows with NO line breaks in the XML content:
<care_plan>{{"diagnosis":{{"primary":"primary diagnosis","secondary":[]}},"treatment_plan":[],"monitoring_plan":[],"follow_up":[],"emergency_plan":"emergency instructions"}}</care_plan>
"""