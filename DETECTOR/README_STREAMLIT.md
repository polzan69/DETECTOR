# Face Detection Streamlit App

This is a Streamlit application for face detection using the SSD ResNet model.

## Features

- Live face detection using your webcam
- Upload and process images for face detection
- Adjust detection confidence threshold
- Display individual detected faces

## Setup Instructions

1. **Install Required Packages**:

   ```bash
   pip install -r streamlit_requirements.txt
   ```

2. **Run the Streamlit App**:

   ```bash
   streamlit run streamlit_app.py
   ```

   This will start the Streamlit server and open your default web browser with the application.

## Hosting on Streamlit Community Cloud

To host this application on Streamlit Community Cloud:

1. Sign up for a free [Streamlit Community Cloud](https://streamlit.io/cloud) account
2. Connect your GitHub repository containing this code
3. Select the `streamlit_app.py` file as your main app file
4. Deploy the application

You will receive a public URL that you can share with others to access your face detection application.

## Notes

- The webcam functionality will only work if the user allows camera access in their browser
- The application uses the SSD ResNet model located in the `DETECTOR/face_detector` directory
- Hosting on Streamlit Community Cloud is completely free and doesn't require a credit card 