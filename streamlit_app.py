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

# Detect if running in Streamlit Cloud
is_cloud = os.environ.get('STREAMLIT_SHARING', '') or os.environ.get('STREAMLIT_CLOUD', '')

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🤖",
    layout="wide"
)

# Debug path information - comment these out once working
# st.write(f"Current directory: {os.getcwd()}")
# st.write(f"Files in directory: {os.listdir('.')}")

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

# Function to start the Flask server in a separate process
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