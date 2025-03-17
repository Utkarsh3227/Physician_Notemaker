# SOAP Note Generator

This project is a **SOAP Note Generator** built using **Streamlit** and **Llama 2** (via `ctransformers`). It converts doctor-patient conversations into structured **SOAP (Subjective, Objective, Assessment, Plan) notes**.

## Features
- Uses **Llama 2** for text generation
- Converts medical conversations into structured SOAP notes
- Outputs the results in **JSON format**
- Simple UI built with **Streamlit**
- Supports GPU acceleration for faster inference

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Install Dependencies
```bash
pip install streamlit ctransformers torch
```

### Download the Model
The Llama 2 model file (`llama-2-7b-chat.Q6_K.gguf`) can be downloaded from **Google Drive**:
[Download Here](YOUR_GOOGLE_DRIVE_LINK)

Place the downloaded model file in the appropriate directory, as specified in the script.

## Usage
### Run the App
```bash
streamlit run app.py
```

### How It Works
1. Enter a conversation between a **doctor and a patient** in the text box.
2. Click **Generate SOAP Note**.
3. The structured **SOAP note** will be displayed in JSON format.

## Project Structure
```
📂 SOAP-Note-Generator
├── app.py                # Main Streamlit app
├── README.md             # Documentation
└── requirements.txt      # List of dependencies
```

## Example Output
### Input (Conversation)
```
Doctor: What brings you in today?
Patient: I've had a persistent cough for the past week.
Doctor: Any fever or shortness of breath?
Patient: No fever, but I feel a little tired.
```

### Output (JSON)
```json
{
  "Subjective": {
    "Chief_Complaint": "Persistent cough for the past week",
    "History_of_Present_Illness": "No fever, mild fatigue."
  },
  "Objective": {
    "Physical_Exam": "Not provided",
    "Observations": "Not provided"
  },
  "Assessment": {
    "Diagnosis": "Possible viral infection",
    "Severity": "Mild"
  },
  "Plan": {
    "Treatment": "Rest, fluids, over-the-counter cough syrup",
    "Follow-Up": "Return if symptoms worsen"
  }
}
```

## License
This project is licensed under the MIT License.

## Contributing
Feel free to fork the repo and submit a pull request if you have improvements!

## Contact
For any issues or suggestions, please open an **issue** in the repository.

