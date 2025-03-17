import json
import re
import streamlit as st
from ctransformers import AutoModelForCausalLM

# Load the model with GPU acceleration
model_path = "/home/dse/Desktop/CD/llama-2-7b-chat.Q6_K.gguf"
model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", gpu_layers=0)

def extract_soap_note(conversation):
    """Generates a SOAP note from a conversation using Llama and converts it into structured JSON."""

    # Prompt to guide the model
    prompt = (
        "Convert the following conversation into a structured SOAP note.\n"
        "Follow this structure:\n\n"
        "Subjective:\n- Chief Complaint:\n- History of Present Illness:\n\n"
        "Objective:\n- Physical Exam:\n- Observations:\n\n"
        "Assessment:\n- Diagnosis:\n- Severity:\n\n"
        "Plan:\n- Treatment:\n- Follow-Up:\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Generate the SOAP note in the above format."
    )

    # Generate response
    response = model(prompt, max_new_tokens=512, temperature=0.3)
    generated_text = response.strip()

    # Convert structured text into JSON format
    structured_output = parse_to_json(generated_text)

    return structured_output

def parse_to_json(text):
    """Parses the generated SOAP note into structured JSON format."""
    default_value = "Not provided"

    # Extract SOAP note using section delimiters
    match = re.search(r"---SOAP Note Start---(.*?)---SOAP Note End---", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    sections = ["Subjective", "Objective", "Assessment", "Plan"]
    fields = {
        "Subjective": ["Chief Complaint", "History of Present Illness"],
        "Objective": ["Physical Exam", "Observations"],
        "Assessment": ["Diagnosis", "Severity"],
        "Plan": ["Treatment", "Follow-Up"]
    }

    structured_data = {section: {} for section in sections}

    for section in sections:
        for field in fields[section]:
            # Use a non-greedy match to prevent over-capturing
            match = re.search(rf"{field}:\s*(.*?)(?:\n[A-Z]|$)", text, re.DOTALL)
            if match:
                cleaned_text = re.sub(r"\*", "", match.group(1)).strip()  # Remove asterisks
                structured_data[section][field.replace(" ", "_")] = cleaned_text
            else:
                structured_data[section][field.replace(" ", "_")] = default_value

    return structured_data

# Streamlit App
st.title("SOAP Note Generator")
st.write("Enter a conversation between a doctor and a patient to generate a structured SOAP note.")

# Input text area for conversation
conversation = st.text_area("Conversation", height=200)

# Button to generate SOAP note
if st.button("Generate SOAP Note"):
    if conversation.strip():
        # Generate SOAP note
        soap_note = extract_soap_note(conversation)
        
        # Display the structured SOAP note
        st.subheader("Generated SOAP Note (JSON):")
        st.json(soap_note)
    else:
        st.warning("Please enter a conversation to generate a SOAP note.")