import streamlit as st
import cv2
import numpy as np
import time
import os
import requests
import threading
from datetime import datetime
import base64
import io
import subprocess
import socket
from PIL import Image

# Detect if running in Streamlit Cloud - force to True if we detect we're in the cloud
is_cloud = os.environ.get('STREAMLIT_SHARING', '') or os.environ.get('STREAMLIT_CLOUD', '')
is_cloud = True if '/mount/src/detector' in os.getcwd() else is_cloud

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🤖",
    layout="wide"
)

# Debug path information - comment these out once working
# st.write(f"Current directory: {os.getcwd()}")
# st.write(f"Files in directory: {os.listdir('.')}")

# Define paths for model files
FACE_DETECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_detector')

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

# Function to check if a port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Function to get local IP address
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# Function to start the Flask server in a separate process (local mode only)
def start_flask_server():
    try:
        # Check if server is already running
        if is_port_in_use(5000):
            st.success("Face detection server is already running on port 5000")
            return True
        
        # Start the Flask server as a subprocess
        cmd = ["python", "face_detection_improved.py"]
        flask_process = subprocess.Popen(cmd)
        
        # Wait for server to start
        start_time = time.time()
        while not is_port_in_use(5000) and time.time() - start_time < 10:
            time.sleep(0.5)
        
        if is_port_in_use(5000):
            st.success("Face detection server started successfully")
            return True
        else:
            st.error("Timed out waiting for face detection server to start")
            return False
    except Exception as e:
        st.error(f"Error starting face detection server: {e}")
        return False

# Function to stop the Flask server
def stop_flask_server():
    try:
        requests.get("http://localhost:5000/shutdown")
        st.success("Face detection server stopped")
    except:
        st.warning("Could not gracefully stop server. It may have already been stopped.")

# Main Streamlit app
def main():
    st.title("Face Detection Application")
    
    # Choose UI based on environment
    if is_cloud:
        cloud_mode_ui()
    else:
        local_mode_ui()

# Streamlit Cloud mode - uses Streamlit components directly
def cloud_mode_ui():
    st.header("Face Detection (Cloud Mode)")
    
    st.warning("""
    **Running in Streamlit Cloud mode.**
    
    In this environment, you can upload and process images, but the live camera streaming functionality is not available.
    
    To use the live camera streaming:
    1. Download the code from GitHub
    2. Run it locally on your computer
    3. Use the server controls to start the streaming server
    """)
    
    # Add a link to the GitHub repository
    st.markdown("""
    ### Download the Full Code
    Get the complete code from the GitHub repository to run the streaming server locally:
    
    [GitHub Repository](https://github.com/YOUR_USERNAME/detector)
    """)
    
    # Sidebar with options
    st.sidebar.title("Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.1)
    
    # Load detector
    detector = load_face_detector(detection_confidence)
    
    # Use upload mode for cloud
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

# Local mode - can run the Flask server
def local_mode_ui():
    # Server control section
    st.header("Face Detection Server Control")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Face Detection Server"):
            with st.spinner("Starting server..."):
                start_flask_server()
    
    with col2:
        if st.button("Stop Face Detection Server"):
            with st.spinner("Stopping server..."):
                stop_flask_server()
    
    # Get local IP address
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:5000"
    
    # Display stream options
    st.header("Live Stream Access")
    st.info(f"Your face detection server can be accessed at: {server_url}")
    
    # Create tabs for different view options
    tab1, tab2 = st.tabs(["Stream in Streamlit", "Access Instructions"])
    
    with tab1:
        # Try to embed the stream directly in Streamlit using an iframe
        if is_port_in_use(5000):
            st.subheader("Live Face Detection Stream")
            st.markdown(f"""
            <iframe src="{server_url}/mobile_stream" width="100%" height="480" allow="camera">
            </iframe>
            """, unsafe_allow_html=True)
            
            # Also display the current image as a fallback
            st.subheader("Current Frame (Refreshes Every 3 Seconds)")
            image_placeholder = st.empty()
            
            # Auto-refresh the current image
            def update_image():
                while True:
                    try:
                        response = requests.get(f"{server_url}/current_image.jpg")
                        if response.status_code == 200:
                            image = Image.open(io.BytesIO(response.content))
                            image_placeholder.image(image, use_column_width=True)
                    except:
                        pass
                    time.sleep(3)
            
            # Start image updater in a thread
            if 'updater_thread' not in st.session_state:
                st.session_state.updater_thread = threading.Thread(target=update_image, daemon=True)
                st.session_state.updater_thread.start()
        else:
            st.warning("Face detection server is not running. Start the server to view the stream.")
    
    with tab2:
        st.subheader("How to Access the Face Detection Stream")
        st.markdown("""
        ### From a Web Browser
        Open any web browser and go to:
        """)
        st.code(server_url)
        
        st.markdown("""
        ### From Your Mobile App (WebView)
        Use the following URL in your WebView component:
        """)
        st.code(f"{server_url}/mobile_stream")
        
        st.markdown("""
        ### API Endpoints
        - **Main interface**: `{server_url}/`
        - **Mobile-optimized stream**: `{server_url}/mobile_stream`
        - **Raw video feed**: `{server_url}/video_feed`
        - **Current image**: `{server_url}/current_image.jpg`
        - **Server status**: `{server_url}/api/status`
        """)

# Flask server shutdown endpoint
# NOTE: This is just documentation for face_detection_improved.py
# The code below is not executed in this file
"""
@app.route('/shutdown')
def shutdown_server():
    # Add this to face_detection_improved.py to allow remote shutdown
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'
"""

if __name__ == "__main__":
    main() 