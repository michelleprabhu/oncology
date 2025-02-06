import streamlit as st
import pandas as pd
import aiohttp
import json
import os
import time
import asyncio
import openai  # Using OpenAI API directly

# Function to validate NCT ID format
def validate_nct_id(nct_id):
    return (
        isinstance(nct_id, str) and 
        nct_id.startswith('NCT') and 
        len(nct_id) == 11 and 
        nct_id[3:].isdigit()
    )

# Function to fetch trial criteria from ClinicalTrials.gov API
async def fetch_trial_criteria(nct_id, session):
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    params = {"format": "json", "markupFormat": "markdown"}
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(api_url, headers=headers, params=params, timeout=15) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return data.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria')
    except Exception:
        return None

# Function to parse criteria text using OpenAI API
async def parse_criteria(criteria_text):
    if not criteria_text or len(criteria_text.strip()) < 50:
        return {"inclusion": [], "exclusion": []}
    prompt = f"""Convert this clinical trial criteria into JSON format:
    {{"inclusion": ["list", "of", "criteria"], "exclusion": ["list", "of", "criteria"]}}
    Input: {criteria_text}
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use GPT-3.5 or another model
            prompt=prompt,
            max_tokens=150
        )
        result = response.choices[0].text.strip()
        parsed = json.loads(result)
        return parsed if isinstance(parsed, dict) else {"inclusion": [], "exclusion": []}
    except Exception as e:
        st.error(f"Error parsing criteria: {e}")
        return {"inclusion": [], "exclusion": []}

# Function to correlate patients with trials
async def correlate_patient_with_trial(patient_info, criterion):
    prompt = f"""Does the patient match the criterion?
    **Patient Info:**
    - Primary Diagnosis: {patient_info['primary_diagnosis']}
    - Secondary Diagnosis: {patient_info['secondary_diagnosis']}
    - Prescription: {patient_info['prescription']}
    - JCode: {patient_info['jcode']}
    **Criterion:** {criterion}
    Answer with 'Yes' or 'No'.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use GPT-3.5 or another model
            prompt=prompt,
            max_tokens=10
        )
        return "Yes" if "yes" in response.choices[0].text.lower() else "No"
    except Exception as e:
        st.error(f"Error correlating patient: {e}")
        return "No"

st.set_page_config(page_title="Patient Trial Eligibility Checker", page_icon="🩺", layout="wide")
col1, col2 = st.columns([1, 6])
with col1:
    st.image("Mool.png", width=150)
with col2:
    st.markdown("<h1>Patient Trial Eligibility Checker</h1>", unsafe_allow_html=True)

with st.spinner("Authenticating Mool AI agent..."):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("API_KEY not found.")
        st.stop()
    openai.api_key = openai_api_key  # Set OpenAI API key
    time.sleep(3)
st.success("Authentication Successful")

uploaded_files = st.file_uploader("Upload files", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
if len(uploaded_files) >= 3:
    clinical_trial_file, patient_database_file, icd_codes_file = uploaded_files[:3]
    trial_df = pd.read_excel(clinical_trial_file, engine='openpyxl', dtype=str)
    patient_df = pd.read_excel(patient_database_file, engine='openpyxl', dtype=str)
    icd_codes_df = pd.read_excel(icd_codes_file, engine='openpyxl', dtype=str)
    icd_dict = dict(zip(icd_codes_df['ICD Code'], icd_codes_df['Disease Name']))
    
    selected_patient = st.selectbox("Select Patient Name", patient_df['Patient Name'].tolist())

    if st.button("Check Eligibility"):
        st.write("Checking eligibility...")
        
        async def process_trials():
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_trial_criteria(nct, session) for nct in trial_df['NCT Number'].tolist()]
                return await asyncio.gather(*tasks)
        
        async def evaluate_patient():
            criteria_data = await process_trials()
            eligibility_results = []
            
            for nct_id, criteria_text in zip(trial_df['NCT Number'], criteria_data):
                if criteria_text:
                    parsed_criteria = await parse_criteria(criteria_text)
                    patient_info = {
                        'primary_diagnosis': icd_dict.get(patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]['Primary Diagnosis'], 'Unknown'),
                        'secondary_diagnosis': icd_dict.get(patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]['Secondary Diagnosis'], 'Unknown'),
                        'prescription': patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]['Prescription'],
                        'jcode': patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]['JCode']
                    }
                    
                    matches = sum(
                        await correlate_patient_with_trial(patient_info, c) == "Yes"
                        for c in parsed_criteria['inclusion']
                    )
                    
                    score = (matches / max(len(parsed_criteria['inclusion']), 1)) * 100
                    eligibility_results.append({
                        'Patient Name': selected_patient,
                        'NCT Number': nct_id,
                        'Eligibility Score': score,
                        'Matched Criteria': matches
                    })
            
            st.write("### Eligibility Summary:")
            st.dataframe(pd.DataFrame(eligibility_results))
        
        asyncio.run(evaluate_patient())
