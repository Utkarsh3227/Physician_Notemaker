import json
import re
import streamlit as st
from transformers import pipeline
from langchain_community.llms import CTransformers

# -----------------------
# Load the NER Model
# -----------------------

ner_pipe = pipeline(
    "token-classification",
    model="blaze999/Medical-NER",
    aggregation_strategy="simple"
)

# Load Llama GGUF model
model_path = "/home/dse/Desktop/CD/llama-2-7b-chat.Q6_K.gguf"
llama_model = CTransformers(
    model=model_path,
    model_type="llama",
    config={"gpu_layers": 0}
)

# -----------------------
# Function Definitions
# -----------------------

def extract_medical_entities(text):
    """Extracts symptoms, diagnosis, and treatment using NER."""
    entities = ner_pipe(text)

    anatomical_structures = []
    symptom_keywords = []
    diagnosis_subs = ["DISEASE_DISORDER"]
    treatment_subs = ["THERAPEUTIC_PROCEDURE", "MEDICATION"]

    medical_summary = {
        "Patient_Name": "Janet Jones",
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

    medical_summary["Symptoms"] = list(set([f"{s.capitalize()} pain" for s in anatomical_structures] + symptom_keywords))
    medical_summary["Treatment"] = list(set(medical_summary["Treatment"]))

    return medical_summary


def refine_text_with_llama(ner_output):
    """Refines extracted medical data using Llama."""
    
    ner_text = (
        f"Patient Name: {ner_output['Patient_Name']}\n"
        f"Symptoms: {', '.join(ner_output['Symptoms'])}\n"
        f"Diagnosis: {ner_output['Diagnosis'] if ner_output['Diagnosis'] else 'None'}\n"
        f"Treatment: {', '.join(ner_output['Treatment']) if ner_output['Treatment'] else 'None'}\n\n"
        "Refine the symptoms by merging redundant expressions. Provide the response in normal text format."
    )

    return llama_model(ner_text)


def parse_llama_output(output_text):
    """Parses Llama refined text back into JSON format."""
    lines = output_text.strip().split("\n")

    if not lines or len(lines) < 1:
        raise ValueError("LLM output is empty or does not contain expected format")

    parsed_data = {}

    for line in lines:
        if ": " in line:  
            key_value = line.split(": ", 1)  
            if len(key_value) == 2:
                key, value = key_value
                parsed_data[key.strip()] = value.strip()

    return parsed_data


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
    """Parses Llama's prognosis output into structured JSON format."""
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
# Streamlit UI
# -----------------------

st.title("🩺 Physician Notemaker - Medical NER & Summarization")

st.write("""
This tool extracts medical information from conversations using Named Entity Recognition (NER) and refines it with Llama-2.
It also provides a **patient's current status and prognosis.**
""")

conversation_text = st.text_area("Enter Medical Conversation:", height=200, 
                                 placeholder="Doctor: How are you feeling today?\nPatient: I had a car accident. My neck and back hurt...")

if st.button("Generate Summary"):
    if conversation_text.strip():
        with st.spinner("Processing..."):
            # Run NER Extraction
            ner_output = extract_medical_entities(conversation_text)

            # Refine using Llama
            refined_text = refine_text_with_llama(ner_output)

            # Parse refined output
            final_medical_summary = parse_llama_output(refined_text)

            # Extract Prognosis and Status
            prognosis_summary = extract_prognosis_status(conversation_text)

            # Merge prognosis and status into final JSON
            final_medical_summary.update(prognosis_summary)

            # Display Results
            st.subheader("📄 Extracted Medical Summary")
            st.json(final_medical_summary)
    else:
        st.warning("Please enter a medical conversation before generating the summary.")

st.markdown("---")
st.markdown("💡 **Developed with Llama-2 and Streamlit** | 🚀 Powered by Transformers & LangChain")



# import json
# import re
# import streamlit as st
# from transformers import pipeline
# from langchain.llms import CTransformers

# # -----------------------
# # PART 1: NER Extraction
# # -----------------------

# # Load the NER model
# @st.cache_resource
# def load_ner_model():
#     return pipeline(
#         "token-classification",
#         model="blaze999/Medical-NER",
#         aggregation_strategy="simple"
#     )

# ner_pipe = load_ner_model()

# def extract_medical_entities(text):
#     """Extracts symptoms, diagnosis, and treatment from the conversation using NER."""
#     entities = ner_pipe(text)

#     anatomical_structures = []
#     symptom_keywords = []
#     diagnosis_subs = ["DISEASE_DISORDER"]
#     treatment_subs = ["THERAPEUTIC_PROCEDURE", "MEDICATION"]

#     medical_summary = {
#         "Patient_Name": "Janet Jones",  # Default name, can be customized
#         "Symptoms": [],
#         "Diagnosis": None,
#         "Treatment": [],
#     }

#     for entity in entities:
#         label = entity["entity_group"].upper()
#         word = entity["word"].strip()

#         if "BIOLOGICAL_STRUCTURE" in label:
#             anatomical_structures.append(word)
#         elif "SIGN_SYMPTOM" in label:
#             symptom_keywords.append(word)
#         elif any(d in label for d in diagnosis_subs):
#             medical_summary["Diagnosis"] = word
#         elif any(t in label for t in treatment_subs):
#             medical_summary["Treatment"].append(word)

#     # Merge anatomical structures with symptoms
#     medical_summary["Symptoms"] = list(set([f"{s.capitalize()} pain" for s in anatomical_structures] + symptom_keywords))
#     medical_summary["Treatment"] = list(set(medical_summary["Treatment"]))

#     return medical_summary

# # -----------------------
# # PART 2: Llama Refinement (Normal Text)
# # -----------------------

# # Load Llama GGUF model
# @st.cache_resource
# def load_llama_model():
#     model_path = "/home/dse/Desktop/CD/llama-2-7b-chat.Q6_K.gguf"
#     return CTransformers(
#         model=model_path,
#         model_type="llama",
#         config={"gpu_layers": 0}
#     )

# llama_model = load_llama_model()

# def refine_medical_summary(ner_output):
#     """Refines the extracted medical summary using Llama."""
#     ner_text = (
#         f"Patient Name: {ner_output['Patient_Name']}\n"
#         f"Symptoms: {', '.join(ner_output['Symptoms'])}\n"
#         f"Diagnosis: {ner_output['Diagnosis'] if ner_output['Diagnosis'] else 'None'}\n"
#         f"Treatment: {', '.join(ner_output['Treatment']) if ner_output['Treatment'] else 'None'}\n\n"
#         "Refine the symptoms by merging redundant expressions (e.g., 'Neck hurt' and 'Back hurt' should be 'Neck and Back pain'). "
#         "Provide the response in normal text format."
#     )

#     # Generate refined text using Llama
#     refined_text = llama_model(ner_text, temperature=0.3, max_new_tokens=512)
#     return refined_text

# def parse_llama_output(text):
#     """Parses Llama refined text back into JSON format with error handling."""
#     default_value = "Not provided"
#     structured_data = {
#         "Patient_Name": default_value,
#         "Symptoms": [],
#         "Diagnosis": default_value,
#         "Treatment": []
#     }

#     lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

#     for line in lines:
#         if line.startswith("Patient Name:"):
#             try:
#                 structured_data["Patient_Name"] = line.split(": ", 1)[1].strip()
#             except IndexError:
#                 pass

#         elif line.startswith("Symptoms:"):
#             try:
#                 symptoms = line.split(": ", 1)[1].strip()
#                 structured_data["Symptoms"] = [s.strip() for s in symptoms.split(",")]
#             except IndexError:
#                 pass

#         elif line.startswith("Diagnosis:"):
#             try:
#                 structured_data["Diagnosis"] = line.split(": ", 1)[1].strip()
#             except IndexError:
#                 pass

#         elif line.startswith("Treatment:"):
#             try:
#                 treatment = line.split(": ", 1)[1].strip()
#                 structured_data["Treatment"] = [t.strip() for t in treatment.split(",")]
#             except IndexError:
#                 pass

#     return structured_data

# # -----------------------
# # PART 3: Extract Current Status & Prognosis
# # -----------------------

# def extract_prognosis_status(conversation):
#     """Extracts prognosis and current status from a medical conversation using Llama."""
#     prompt = (
#         "Analyze the following medical conversation and extract the patient's:\n\n"
#         "- Current Status: A short summary of the patient's present condition.\n"
#         "- Prognosis: An expected outlook or future condition based on the conversation.\n\n"
#         f"Conversation:\n{conversation}\n\n"
#         "Generate the response in the format:\n"
#         "Current Status: <summary>\n"
#         "Prognosis: <expected outcome>\n"
#     )

#     # Generate response
#     response = llama_model(prompt, temperature=0.3, max_new_tokens=512)
#     generated_text = response.strip()

#     # Convert structured text into JSON format
#     return parse_prognosis_output(generated_text)

# def parse_prognosis_output(text):
#     """Parses Llama's prognosis output into structured JSON format."""
#     default_value = "Not provided"

#     match = re.search(r"Current Status:\s*(.*?)(?:\nPrognosis:|$)", text, re.DOTALL)
#     current_status = match.group(1).strip() if match else default_value

#     match = re.search(r"Prognosis:\s*(.*)", text, re.DOTALL)
#     prognosis = match.group(1).strip() if match else default_value

#     return {
#         "Current_Status": current_status,
#         "Prognosis": prognosis
#     }

# # -----------------------
# # Streamlit App
# # -----------------------

# st.title("Medical Summary Generator")
# st.write("Enter a conversation between a doctor and a patient to generate a structured medical summary.")

# # Input text area for conversation
# conversation = st.text_area("Conversation", height=200)

# # Button to generate medical summary
# if st.button("Generate Medical Summary"):
#     if conversation.strip():
#         # Step 1: Extract medical entities using NER
#         ner_output = extract_medical_entities(conversation)

#         # Step 2: Refine the extracted data using Llama
#         refined_text = refine_medical_summary(ner_output)
#         final_medical_summary = parse_llama_output(refined_text)

#         # Step 3: Extract prognosis and current status
#         prognosis_summary = extract_prognosis_status(conversation)
#         final_medical_summary.update(prognosis_summary)

#         # Display the final structured medical summary
#         st.subheader("Final Structured Medical Summary:")
#         st.json(final_medical_summary)
#     else:
#         st.warning("Please enter a conversation to generate a medical summary.")