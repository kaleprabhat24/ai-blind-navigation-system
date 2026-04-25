import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import time
import base64
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration & Caching
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Navigation Assistant", page_icon="🚶‍♂️", layout="wide")

@st.cache_resource
def load_model():
    # Cache the model so it doesn't reload on every UI interaction
    return YOLO("yolov8n.pt") 

model = load_model()

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def estimate_distance(box, class_id):
    """Estimate distance using focal length and bounding box width."""
    x1, y1, x2, y2 = box
    pixel_width = x2 - x1

    # Assuming class 0 is person, others (like 56 for chair) are objects
    real_width = 0.5 if class_id == 0 else 0.7
    focal_length = 700
    
    if pixel_width == 0:
        return 0.0
        
    distance = (real_width * focal_length) / pixel_width
    return distance

def get_navigation(cx, width):
    """Determine navigation instruction based on horizontal position."""
    left_bound = width / 3
    right_bound = (2 * width) / 3
    
    if cx < left_bound:
        return "Obstacle on left, please move right"
    elif cx > right_bound:
        return "Obstacle on right, please move left"
    else:
        return "Obstacle in center, please stop"

def create_audio_player(text):
    """Generates TTS and returns a hidden HTML audio player for autoplay."""
    if not text.strip():
        return ""
        
    filename = "voice.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    
    # Encode to base64 to play directly in HTML
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        
    # FIX: Wrap in a completely unique div so Streamlit unmounts and remounts 
    # the element. This forces the browser's 'autoplay' to trigger every time.
    unique_id = int(time.time() * 1000)
    md = f"""
        <div id="audio_container_{unique_id}">
            <audio autoplay="autoplay" style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        </div>
        """
    return md

# ---------------------------------------------------------
# 3. Main Streamlit UI
# ---------------------------------------------------------
st.markdown("# 🚶‍♂️ Smart Navigation Assistant")
st.markdown("Detects people and chairs, estimates distance, and provides audio navigation cues.")

tab1, tab2 = st.tabs(["🔴 Live Webcam", "📁 Upload Image"])

# --- TAB 1: LIVE WEBCAM ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run_webcam = st.toggle("Start Live Webcam", key="webcam_toggle")
        frame_window = st.empty()
        
    with col2:
        st.markdown("### Navigation Status")
        status_text = st.empty()
        audio_player = st.empty()
        
    if run_webcam:
        cap = cv2.VideoCapture(0)
        
        # State variables for continuous loop
        last_process_time = 0.0
        audio_unlock_time = 0.0       # Tracks when the current sentence will finish
        last_nav_state = ""           # Tracks the last location state to prevent repeating
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the webcam.")
                break
                
            current_time = time.time()
            
            # Run detection every 0.5 seconds to keep visuals responsive
            if current_time - last_process_time >= 0.5:
                results = model(frame, classes=[0, 56])
                
                output_text = ""
                current_nav_states = [] # Collect all navigation directions in this frame
                
                if len(results[0].boxes) == 0:
                    output_text = "Path is clear."
                    current_nav_states.append("clear")
                else:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    h, w, _ = frame.shape
                    
                    for box, cls in zip(boxes, classes):
                        cx = (box[0] + box[2]) / 2
                        dist = estimate_distance(box, int(cls))
                        nav = get_navigation(cx, w)
                        
                        label = model.names[int(cls)]
                        output_text += f"{label} detected at {dist:.1f} meters. {nav}.\n"
                        current_nav_states.append(nav)
                    
                    frame = results[0].plot()
                
                # Update visual text status instantly
                status_text.info(output_text)
                
                # --- AUDIO LOGIC ---
                # Create a string representing the current physical location state
                current_nav_string = " | ".join(current_nav_states)
                
                # 1. Check if we are allowed to speak (previous sentence has finished)
                if current_time >= audio_unlock_time:
                    
                    # 2. Check if the object moved to a NEW location (or if path cleared)
                    if current_nav_string != last_nav_state:
                        
                        # FIX: Clear the previous audio element before injecting the new one
                        audio_player.empty() 

                        # Generate and play audio
                        audio_html = create_audio_player(output_text)
                        audio_player.markdown(audio_html, unsafe_allow_html=True)
                        
                        # Calculate how long this sentence will take to speak
                        # (Average human speech TTS is ~15 characters per second)
                        estimated_speech_time = (len(output_text) / 15.0) + 1.0 
                        
                        # Lock the audio channel until this sentence is done
                        audio_unlock_time = current_time + estimated_speech_time
                        
                        # Update the state so it doesn't repeat this location again
                        last_nav_state = current_nav_string

                last_process_time = current_time

            # Feed the video seamlessly (no flickering)
            frame_window.image(frame, channels="BGR", use_container_width=True)
            
        cap.release()
    else:
        status_text.info("Webcam is currently inactive. Toggle the button to start.")

# --- TAB 2: UPLOAD IMAGE ---
with tab2:
    uploaded_file = st.file_uploader("Upload a static image for analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col3, col4 = st.columns([2, 1])
        
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with col3:
            st.markdown("### Processed Image")
            img_placeholder = st.empty()
            img_placeholder.image(frame, channels="BGR", use_container_width=True)
            
        with col4:
            st.markdown("### Navigation Status")
            with st.spinner("Analyzing image..."):
                results = model(frame, classes=[0, 56])
                
                if len(results[0].boxes) == 0:
                    final_status = "Path is clear."
                else:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    h, w, _ = frame.shape
                    
                    final_status = ""
                    for box, cls in zip(boxes, classes):
                        cx = (box[0] + box[2]) / 2
                        dist = estimate_distance(box, int(cls))
                        nav = get_navigation(cx, w)
                        label = model.names[int(cls)]
                        final_status += f"{label} detected at {dist:.1f} meters. {nav}.\n"
                    
                    annotated_frame = results[0].plot()
                    img_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            
            st.info(final_status)
            audio_markup = create_audio_player(final_status)
            st.markdown(audio_markup, unsafe_allow_html=True)