import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
import base64
import io
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🤖",
    layout="wide"
)

# Define paths for model files
DETECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DETECTOR')
FACE_DETECTOR_DIR = os.path.join(DETECTOR_DIR, 'face_detector')

# Standalone DNNFaceDetector class
class DNNFaceDetector:
    """Class to handle SSD ResNet face detection model"""
    def __init__(self, threshold=0.5):
        self.net = None
        self.initialized = False
        self.threshold = threshold
        self.initialize()
    
    def initialize(self):
        """Initialize the DNN model"""
        try:
            model_path = os.path.join(FACE_DETECTOR_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
            config_path = os.path.join(FACE_DETECTOR_DIR, 'deploy.prototxt')
            
            if not os.path.exists(model_path) or not os.path.exists(config_path):
                st.error(f"Model files not found for SSD ResNet. Looking for {model_path}")
                return False
            
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.size = (300, 300)
            self.scale = 1.0
            self.mean = [104.0, 177.0, 123.0]
            self.swapRB = True
        
            # Set computation preferences for better performance
            if self.net is not None:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                st.success("Successfully initialized SSD ResNet face detector")
                self.initialized = True
                return True
            else:
                st.error("Failed to initialize SSD ResNet face detector")
                return False
                
        except Exception as e:
            st.error(f"Error initializing SSD ResNet face detector: {e}")
            return False
    
    def detect(self, frame):
        """Detect faces using the DNN model"""
        if not self.initialized or self.net is None:
            return []
        
        height, width = frame.shape[:2]
        faces = []
        
        try:
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                frame, self.scale, self.size,
                mean=self.mean,
                swapRB=self.swapRB
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Process detections for SSD ResNet
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < self.threshold:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Convert numpy integers to Python integers
                startX = int(startX)
                startY = int(startY)
                endX = int(endX)
                endY = int(endY)
                
                # Ensure coordinates are within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width, endX)
                endY = min(height, endY)
                
                # Skip invalid detections
                if startX >= endX or startY >= endY:
                    continue
                
                # Use standard Python types to avoid JSON serialization issues
                faces.append({
                    'x': int(startX),
                    'y': int(startY),
                    'width': int(endX - startX),
                    'height': int(endY - startY),
                    'confidence': float(confidence)
                })
            
            return faces
            
        except Exception as e:
            st.error(f"Error in face detection: {e}")
            return []

# Initialize face detector
@st.cache_resource
def load_face_detector(threshold=0.5):
    detector = DNNFaceDetector(threshold)
    return detector

# Function to process a frame for face detection
def process_frame(frame, detector):
    # Detect faces
    faces = detector.detect(frame)
    
    # Draw rectangles around faces
    result_frame = frame.copy()
    for face in faces:
        x, y, w, h = face['x'], face['y'], face['width'], face['height']
        confidence = face['confidence']
        
        # Draw rectangle
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw confidence
        text = f"{confidence:.2f}"
        cv2.putText(result_frame, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_frame, faces

# Main Streamlit app
def main():
    st.title("Face Detection Application")
    
    # Sidebar with options
    st.sidebar.title("Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.1)
    
    # Load detector
    detector = load_face_detector(detection_confidence)
    
    # Camera input or file upload option
    option = st.sidebar.radio("Input Source", ["Camera", "Upload Image"])
    
    if option == "Camera":
        st.header("Live Camera Feed")
        
        # Start/stop camera buttons
        col1, col2 = st.columns(2)
        with col1:
            start_camera = st.button("Start Camera")
        with col2:
            stop_button = st.button("Stop Camera")
        
        # Create a placeholder for the video feed
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Session state to track camera state
        if 'stop_camera' not in st.session_state:
            st.session_state.stop_camera = False
            
        if stop_button:
            st.session_state.stop_camera = True
            
        if start_camera:
            st.session_state.stop_camera = False
            
            # Open the webcam
            cap = cv2.VideoCapture(0)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                st.error("Error: Could not open camera.")
                return
            
            while cap.isOpened() and not st.session_state.stop_camera:
                # Read a frame
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture image")
                    break
                
                # Process the frame
                processed_frame, faces = process_frame(frame, detector)
                
                # Convert color from BGR to RGB
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                # Show detection info
                info_text = f"Detected {len(faces)} faces"
                info_placeholder.info(info_text)
                
                # Add a small delay
                time.sleep(0.1)
            
            # Release the camera
            cap.release()
    
    else:  # Upload Image option
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process the image
            processed_image, faces = process_frame(image, detector)
            
            # Convert color from BGR to RGB
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Display the image
            st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
            
            # Show detection results
            st.subheader(f"Detection Results: {len(faces)} faces found")
            
            # Display each face separately
            if len(faces) > 0:
                st.write("Detected Faces:")
                
                # Create columns for faces
                cols = st.columns(min(len(faces), 4))
                
                for i, face in enumerate(faces):
                    x, y, w, h = face['x'], face['y'], face['width'], face['height']
                    confidence = face['confidence']
                    
                    # Extract face ROI
                    face_roi = image[y:y+h, x:x+w]
                    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    
                    # Display in column
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        st.image(face_roi_rgb, caption=f"Face {i+1} ({confidence:.2f})")

if __name__ == "__main__":
    main() 