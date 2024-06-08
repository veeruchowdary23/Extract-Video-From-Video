import os
import tensorflow as tf
import warnings
import cv2
import mediapipe as mp
import numpy as np
import pytesseract

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Tesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  # Adjust the path to your Tesseract executable

# Open the video file or capture device
video_path = 'Video1.mp4'  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

print(f"Successfully opened video file {video_path}")

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
output_path = 'output_with_human_detection_for_video1.mp4'  # Change this to your desired output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create a VideoWriter object for the extracted person video
person_output_path = 'person_extracted1.mp4'  # Change this to your desired output file path
person_out = cv2.VideoWriter(person_output_path, fourcc, fps/3, (width, height))

# Function to detect the color red in a given frame
def detect_red_jersey(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    return mask

# Function to extract and recognize jersey numbers
def recognize_jersey_number(frame, mask):
    # Ensure mask is 8-bit and single-channel
    mask = mask.astype(np.uint8)
    # Apply mask to get the jersey area
    jersey_area = cv2.bitwise_and(frame, frame, mask=mask)
    # Convert to grayscale and apply adaptive threshold
    gray = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Use Tesseract to recognize text
    config = "--psm 7"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

# Function to process each frame
def process_frame(frame):
    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame for better performance
    image_resized = cv2.resize(image_rgb, (640, 360))

    # Process the frame with MediaPipe Holistic
    results = holistic.process(image_resized)

    # Detect red jerseys
    red_mask = detect_red_jersey(frame)
    red_output = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Draw landmarks only if a red jersey is detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))
    
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))

    # Highlight the person with the red jersey
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            # Increase the bounding box size to capture the whole body
            padding = 70  # Adjust this value to include more area around the person
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle around detected red jersey
            
            # Ensure the mask is correctly sized and type
            red_mask_roi = red_mask[y:y+h, x:x+w]
            if red_mask_roi.shape[0] > 0 and red_mask_roi.shape[1] > 0:
                jersey_number = recognize_jersey_number(frame[y:y+h, x:x+w], red_mask_roi)
                if jersey_number:
                    cv2.putText(frame, jersey_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Save the person with the red jersey as a separate video
                    person_frame = frame[y:y+h, x:x+w]
                    person_frame_resized = cv2.resize(person_frame, (width, height))
                    person_out.write(person_frame_resized)
    
    return frame

# Function to read and process frames
def read_and_process_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print("Processing frame...")

        # Process the frame
        frame = process_frame(frame)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (slow down the display to better see the results)
        cv2.imshow('Human Detection', frame)
        cv2.waitKey(100)  # Adjust the delay as needed (50 ms here)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Finished processing frames.")

# Process frames without threading to debug
read_and_process_frames()

# Release resources
cap.release()
out.release()
person_out.release()
cv2.destroyAllWindows()
