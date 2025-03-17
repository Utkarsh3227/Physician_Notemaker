
import json
import re
import argparse
import streamlit as st
from ctransformers import AutoModelForCausalLM

def load_model(model_path):
    """Loads the model from the given path with GPU acceleration."""
    return AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", gpu_layers=0)

def extract_soap_note(conversation, temperature, model):
    """Generates a SOAP note from a conversation using Llama and converts it into structured JSON."""
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
    
    response = model(prompt, max_new_tokens=512, temperature=temperature)
    generated_text = response.strip()
    
    # Convert structured text into JSON format
    return parse_to_json(generated_text)

def parse_to_json(text):
    """Parses the generated SOAP note into structured JSON format."""
    default_value = "Not provided"
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
            match = re.search(rf"{field}:\s*(.*?)(?:\n[A-Z]|$)", text, re.DOTALL)
            if match:
                cleaned_text = re.sub(r"\\*", "", match.group(1)).strip()
                structured_data[section][field.replace(" ", "_")] = cleaned_text
            else:
                structured_data[section][field.replace(" ", "_")] = default_value
    
    return structured_data

def check_for_errors(soap_note):
    """Checks if the generated SOAP note contains only 'Not provided' values."""
    return all(
        all(value == "Not provided" for value in section.values())
        for section in soap_note.values()
    )

def main():
    parser = argparse.ArgumentParser(description="SOAP Note Generator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    
    # Streamlit App
    st.title("SOAP Note Generator")
    st.write("Enter a conversation between a doctor and a patient to generate a structured SOAP note.")
    
    # Input text area for conversation
    conversation = st.text_area("Conversation", height=200)
    
    # Button to generate SOAP note
    if st.button("Generate SOAP Note"):
        if conversation.strip():
            max_attempts = 10
            valid_note = None
            base_temperature = 0.3
            
            for attempt in range(1, max_attempts + 1):
                current_temperature = base_temperature + (attempt - 1) * 0.1
                with st.spinner(f"Generating ..."):
                    soap_note = extract_soap_note(conversation, current_temperature, model)
                    if not check_for_errors(soap_note):
                        valid_note = soap_note
                        break
            
            if valid_note is None:
                st.error("An error occurred while generating the SOAP. Please try again.")
            else:
                st.subheader("Generated SOAP Note (JSON):")
                st.json(valid_note)
        else:
            st.warning("Please enter a conversation to generate a SOAP note.")

if __name__ == "__main__":
    main()