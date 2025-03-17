# Medical Conversation to JSON Converter

This project is a **Medical Conversation to JSON Converter** built using **Streamlit** and **Llama 2** (via `ctransformers`). It extracts key medical details from doctor-patient conversations and structures them into a **JSON format**.

## Features
- Uses **Llama 2** for text generation
- Extracts patient information, symptoms, diagnosis, treatment, and prognosis
- Outputs the results in **structured JSON format**
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
[Download Here](https://drive.google.com/file/d/1NbIqcK00t6wrVCiV_J4SFi6_Uj2w7Hf6/view?usp=drive_link)

Place the downloaded model file in the appropriate directory, as specified in the script.

## Usage
### Run the App
```bash
streamlit run app.py
```

### How It Works
1. Enter a **doctor-patient conversation** in the text box.
2. Click **Generate JSON Output**.
3. The structured **medical details** will be displayed in JSON format.

## Project Structure
```
📂 Medical-Conversation-JSON
├── app.py                # Main Streamlit app
├── README.md             # Documentation
└── requirements.txt      # List of dependencies
```

## Example Output
### Input (Conversation)
```
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
```

### Output (JSON)
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}
```



## Contributing
Feel free to fork the repo and submit a pull request if you have improvements!

## Contact
For any issues or suggestions, please open an **issue** in the repository.

