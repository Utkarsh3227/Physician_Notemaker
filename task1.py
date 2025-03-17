import streamlit as st
import json
import re
import argparse
from transformers import pipeline
from langchain.llms import CTransformers

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the LLaMA model file")
args = parser.parse_args()
MODEL_PATH = args.model_path

# Streamlit UI
st.title("Medical NER Extraction")
st.write("Enter the clinical conversation to extract medical information.")
text = st.text_area("Enter Clinical Conversation:", height=150)

# -----------------------
# 📌 PART 1: NER Extraction
# -----------------------

@st.cache_resource
def load_ner_model():
    return pipeline("token-classification", model="blaze999/Medical-NER", aggregation_strategy="simple")

ner_pipe = load_ner_model()

def extract_medical_entities(text):
    """Extracts symptoms, diagnosis, and treatment from the conversation using NER."""
    entities = ner_pipe(text)
    anatomical_structures = []
    symptom_keywords = []
    diagnosis_subs = ["DISEASE_DISORDER"]
    treatment_subs = ["THERAPEUTIC_PROCEDURE", "MEDICATION"]

    medical_summary = {
        "Symptoms": [],
        "Diagnosis": None,
        "Treatment": [],
    }

    for entity in entities:
        label = entity["entity_group"].upper()
        word = entity["word"].strip()

        if "BIOLOGICAL_STRUCTURE" in label:
            anatomical_structures.append(word)
        elif "SIGN_SYMPTOM" in label:
            symptom_keywords.append(word)
        elif any(d in label for d in diagnosis_subs):
            medical_summary["Diagnosis"] = word
        elif any(t in label for t in treatment_subs):
            medical_summary["Treatment"].append(word)

    # Merge anatomical structures with symptoms
    medical_summary["Symptoms"] = list(set([f"{s.capitalize()} pain" for s in anatomical_structures] + symptom_keywords))
    medical_summary["Treatment"] = list(set(medical_summary["Treatment"]))

    return medical_summary

# -----------------------
# 📌 PART 2: Llama Refinement
# -----------------------

@st.cache_resource
def load_llama_model(model_path):
    return CTransformers(model=model_path, model_type="llama", config={"gpu_layers": 0})

llama_model = load_llama_model(MODEL_PATH)

def is_valid_name(name):
    """Returns True if the name contains only letters, spaces, hyphens, or apostrophes."""
    return bool(re.fullmatch(r"[A-Za-z\s\-']+", name)) and name.strip().lower() != "unknown"

def extract_patient_name(conversation):
    """Extracts the patient's name from the conversation using Llama.
       If no valid name is found, it returns 'Unknown'."""
    prompt = (
        "Analyze the following clinical conversation and extract the patient's name. "
        "If the name is not explicitly mentioned, return 'Unknown'.\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Patient Name: "
    )
    response = llama_model(prompt)
    patient_name = response.strip()
    return patient_name if is_valid_name(patient_name) else "Unknown"

def refine_medical_text(ner_output, patient_name):
    """Refines extracted medical data using Llama model."""
    ner_text = (
        f"Patient Name: {patient_name}\n"
        f"Symptoms: {', '.join(ner_output['Symptoms'])}\n"
        f"Diagnosis: {ner_output['Diagnosis'] if ner_output['Diagnosis'] else 'None'}\n"
        f"Treatment: {', '.join(ner_output['Treatment']) if ner_output['Treatment'] else 'None'}\n\n"
        "Refine the symptoms by merging redundant expressions (e.g., 'Neck hurt' and 'Back hurt' should be 'Neck and Back pain'). "
        "Provide the response in normal text format."
    )
    return llama_model(ner_text)

def parse_llama_output(text):
    """Parses Llama refined text back into JSON format."""
    lines = text.strip().split("\n")
    refined_json = {
        "Patient_Name": lines[0].split(": ")[1].strip(),
        "Symptoms": [s.strip() for s in lines[1].split(": ")[1].split(",")],
        "Diagnosis": lines[2].split(": ")[1].strip(),
        "Treatment": [t.strip() for t in lines[3].split(": ")[1].split(",")]
    }
    return refined_json

# -----------------------
# 📌 PART 3: Extract Prognosis
# -----------------------

def extract_prognosis_status(conversation):
    """Extracts prognosis and current status using Llama."""
    prompt = (
        "Analyze the following medical conversation and extract the patient's:\n\n"
        "- Current Status: A short summary of the patient's present condition.\n"
        "- Prognosis: An expected outlook or future condition based on the conversation.\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Generate the response in the format:\n"
        "Current Status: <summary>\n"
        "Prognosis: <expected outcome>\n"
    )
    response = llama_model(prompt)
    return parse_prognosis_output(response.strip())

def parse_prognosis_output(text):
    """Parses prognosis output into JSON format."""
    default_value = "Not provided"
    match = re.search(r"Current Status:\s*(.*?)(?:\nPrognosis:|$)", text, re.DOTALL)
    current_status = match.group(1).strip() if match else default_value
    match = re.search(r"Prognosis:\s*(.*)", text, re.DOTALL)
    prognosis = match.group(1).strip() if match else default_value
    return {
        "Current_Status": current_status,
        "Prognosis": prognosis
    }

# -----------------------
# 📌 PART 4: Run Extraction & Display Results
# -----------------------

if st.button("Extract Medical Data"):
    if text:
        with st.spinner("Processing..."):
            # Step 1: Extract patient name
            patient_name = extract_patient_name(text)
            # Step 2: Extract medical entities
            extracted_data = extract_medical_entities(text)
            
            # Step 3: Refine extracted data with a retry mechanism
            max_attempts = 10
            refined_data = None
            for attempt in range(1, max_attempts + 1):
                refined_text = refine_medical_text(extracted_data, patient_name)
                try:
                    refined_data = parse_llama_output(refined_text)
                    break  # Exit loop if parsing is successful
                except Exception as e:
                    continue
            
            if refined_data is None:
                st.error("")
                refined_data = {
                    "Patient_Name": patient_name,
                    "Symptoms": extracted_data["Symptoms"],
                    "Diagnosis": extracted_data["Diagnosis"],
                    "Treatment": extracted_data["Treatment"],
                    "Current_Status": "Not provided",
                    "Prognosis": "Not provided"
                }
            # Step 4: Extract prognosis & status
            prognosis_data = extract_prognosis_status(text)
            refined_data.update(prognosis_data)
            
        st.subheader("Structured Medical Summary")
        st.json(refined_data)
    else:
        st.warning("Please enter a clinical conversation!")
